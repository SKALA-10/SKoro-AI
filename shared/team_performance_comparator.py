# team_performance_comparator.py
# 팀 성과 비교 분석 모듈 - 클러스터링 기반 유사팀 식별 및 성과 통계

import pandas as pd
import numpy as np
import json
import re
import statistics
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# 머신러닝 라이브러리
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# DB 연결
from sqlalchemy import create_engine, text
import sys
import os

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(project_root)

from config.settings import DatabaseConfig
from dotenv import load_dotenv

load_dotenv()

# DB 설정
db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


class TeamPerformanceDB:
    """팀 성과 비교 전용 DB 클래스"""
    
    def __init__(self):
        self.engine = engine
    
    def fetch_all_team_kpis(self) -> List[Dict]:
        """전사 모든 팀의 KPI 데이터 조회 (클러스터링용)"""
        with self.engine.connect() as connection:
            query = text("""
                SELECT 
                    tk.team_kpi_id,
                    tk.team_id,
                    tk.kpi_name,
                    tk.kpi_description,
                    t.team_name,
                    h.headquarter_name
                FROM team_kpis tk
                JOIN teams t ON tk.team_id = t.team_id
                JOIN headquarters h ON t.headquarter_id = h.headquarter_id
                WHERE tk.year = 2024
                ORDER BY tk.team_id, tk.team_kpi_id
            """)
            
            results = connection.execute(query).fetchall()
            return [dict(row._mapping) for row in results]
    
    def fetch_team_performance_data(self, team_id: int, period_id: int) -> Optional[Dict]:
        """팀 성과 데이터 조회 (team_evaluations.average_achievement_rate 우선)"""
        with self.engine.connect() as connection:
            query = text("""
                SELECT 
                    te.average_achievement_rate as overall_rate,
                    t.team_name
                FROM team_evaluations te
                JOIN teams t ON te.team_id = t.team_id
                WHERE te.team_id = :team_id AND te.period_id = :period_id
            """)
            
            result = connection.execute(query, {"team_id": team_id, "period_id": period_id}).fetchone()
            
            if result:
                return {
                    "team_id": team_id,
                    "team_name": result.team_name,
                    "overall_rate": result.overall_rate or 0
                }
            return None


class TextPreprocessor:
    """텍스트 전처리 클래스"""
    
    def __init__(self):
        self.stopwords = [
            '의', '를', '을', '이', '가', '에', '는', '은', '과', '와', '로', '으로',
            '에서', '부터', '까지', '에게', '한테', '께', '으며', '며', '하여', '해서',
            '하고', '그리고', '또한', '또는', '그런데', '하지만', '그러나', '따라서',
            '시스템', '프로젝트', '업무', '담당', '수행', '진행', '개발', '관리', '운영'
        ]
    
    def clean_text(self, text: str) -> str:
        """텍스트 정리 (특수문자 제거, 공백 정규화)"""
        if not text:
            return ""
        text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """토큰화 및 불용어 제거"""
        tokens = text.split()
        filtered_tokens = [token for token in tokens 
                          if token not in self.stopwords and len(token) >= 2]
        return filtered_tokens
    
    def preprocess(self, text: str) -> str:
        """전체 전처리 파이프라인"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return ' '.join(tokens)


class TeamClusteringAnalyzer:
    """팀 클러스터링 분석 클래스"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.db = TeamPerformanceDB()
        self.team_data = None
        self.cluster_labels = None
        self.similarity_threshold = 0.2
        
    def load_team_data(self):
        """팀 KPI 데이터 로드 및 전처리"""
        print("팀 KPI 데이터 로드 중...")
        raw_data = self.db.fetch_all_team_kpis()
        
        if not raw_data:
            raise ValueError("팀 KPI 데이터가 없습니다.")
        
        team_texts = {}
        team_info = {}
        
        for row in raw_data:
            team_id = row['team_id']
            kpi_text = f"{row['kpi_name']} {row['kpi_description']}"
            preprocessed_text = self.preprocessor.preprocess(kpi_text)
            
            if team_id not in team_texts:
                team_texts[team_id] = []
                team_info[team_id] = {
                    'team_name': row['team_name'],
                    'headquarter_name': row['headquarter_name']
                }
            
            team_texts[team_id].append(preprocessed_text)
        
        self.team_data = []
        for team_id, texts in team_texts.items():
            combined_text = ' '.join(texts)
            self.team_data.append({
                'team_id': team_id,
                'combined_text': combined_text,
                'team_name': team_info[team_id]['team_name'],
                'headquarter_name': team_info[team_id]['headquarter_name'],
                'kpi_count': len(texts)
            })
        
        print(f"총 {len(self.team_data)}개 팀 데이터 로드 완료")
        return self.team_data
    
    def perform_clustering(self):
        """KMeans 클러스터링 수행"""
        texts = [team['combined_text'] for team in self.team_data]
        
        # TF-IDF 벡터화
        tfidf_vectorizer = TfidfVectorizer(
            max_features=50,
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 1)
        )
        
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        
        # 최적 클러스터 개수 찾기
        n_teams = len(self.team_data)
        max_clusters = min(10, n_teams - 1)
        
        if max_clusters < 2:
            # 팀이 너무 적으면 모든 팀을 하나의 클러스터로
            self.cluster_labels = [0] * n_teams
            for i, team in enumerate(self.team_data):
                team['cluster'] = 0
            print("팀 수가 부족하여 단일 클러스터로 설정")
            return self.cluster_labels
        
        best_score = -1
        best_k = 2
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(tfidf_matrix)
            score = silhouette_score(tfidf_matrix, labels)
            print(f"클러스터 {k}개: Silhouette Score = {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        # 최종 클러스터링
        print(f"최적 클러스터 개수: {best_k} (Score: {best_score:.3f})")
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # 클러스터 결과를 팀 데이터에 저장
        for i, team in enumerate(self.team_data):
            team['cluster'] = int(self.cluster_labels[i])
        
        # 클러스터별 분포 출력
        cluster_counts = {}
        for label in self.cluster_labels:
            cluster_counts[label] = cluster_counts.get(label, 0) + 1
        
        print("클러스터별 팀 분포:")
        for cluster_id, count in sorted(cluster_counts.items()):
            print(f"  클러스터 {cluster_id}: {count}개 팀")
        
        return self.cluster_labels
    
    def get_team_cluster(self, team_id: int) -> int:
        """특정 팀의 클러스터 ID 반환"""
        for team in self.team_data:
            if team['team_id'] == team_id:
                return team['cluster']
        return -1
    
    def get_cluster_teams(self, cluster_id: int) -> List[int]:
        """특정 클러스터의 팀 ID 목록 반환"""
        return [team['team_id'] for team in self.team_data if team['cluster'] == cluster_id]
    
    def get_clusters_mapping(self) -> Dict[int, List[int]]:
        """모든 클러스터의 팀 ID 매핑 반환"""
        clusters = {}
        for team in self.team_data:
            cluster_id = team['cluster']
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(team['team_id'])
        return clusters


class ClusterPerformanceStatsManager:
    """클러스터 성과 통계 관리 클래스"""
    
    def __init__(self, cache_dir="./data/cache"):
        self.cache_dir = cache_dir
        self.db = TeamPerformanceDB()
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_file_path(self, period_id: int) -> str:
        """캐시 파일 경로 생성"""
        return os.path.join(self.cache_dir, f"cluster_performance_Q{period_id}_2024.json")
    
    def check_stats_exists(self, period_id: int) -> bool:
        """클러스터 통계 파일 존재 확인"""
        return os.path.exists(self.get_cache_file_path(period_id))
    
    def load_cluster_stats(self, period_id: int) -> Dict:
        """클러스터 성과 통계 로드"""
        cache_file = self.get_cache_file_path(period_id)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("cluster_stats", {})
            except Exception as e:
                print(f"캐시 파일 로드 실패: {e}")
                return {}
        return {}
    
    def save_cluster_stats(self, cluster_stats: Dict, period_id: int):
        """클러스터 성과 통계 저장"""
        cache_file = self.get_cache_file_path(period_id)
        stats_with_metadata = {
            "cluster_stats": cluster_stats,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "period_id": period_id,
                "total_clusters": len(cluster_stats),
                "total_teams": sum(len(stats["teams"]) for stats in cluster_stats.values())
            }
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(stats_with_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"클러스터 통계 저장 완료: {cache_file}")
    
    def calculate_cluster_performance_stats(self, period_id: int, force_recalculate: bool = False) -> Dict:
        """클러스터별 성과 통계 계산"""
        
        # 기존 캐시 확인
        if not force_recalculate and self.check_stats_exists(period_id):
            print(f"기존 클러스터 통계 사용 (Q{period_id})")
            return self.load_cluster_stats(period_id)
        
        print(f"클러스터별 성과 통계 계산 시작 (Q{period_id})...")
        
        # 1. 팀 클러스터링 수행
        clustering_analyzer = TeamClusteringAnalyzer()
        clustering_analyzer.load_team_data()
        clustering_analyzer.perform_clustering()
        
        # 2. 클러스터별로 그룹화
        clusters = clustering_analyzer.get_clusters_mapping()
        
        # 3. 각 클러스터별 성과 통계 계산
        cluster_stats = {}
        
        for cluster_id, team_ids in clusters.items():
            print(f"클러스터 {cluster_id} 처리 중... ({len(team_ids)}개 팀)")
            
            cluster_team_performances = []
            
            for team_id in team_ids:
                team_perf = self.db.fetch_team_performance_data(team_id, period_id)
                if team_perf:
                    cluster_team_performances.append(team_perf)
            
            if not cluster_team_performances:
                print(f"클러스터 {cluster_id}: 성과 데이터 없음")
                continue
            
            # 전체 팀 달성률 통계
            overall_rates = [team["overall_rate"] for team in cluster_team_performances]
            
            cluster_stats[str(cluster_id)] = {
                "teams": [team["team_id"] for team in cluster_team_performances],
                "team_count": len(cluster_team_performances),
                "overall_stats": {
                    "avg_rate": round(statistics.mean(overall_rates), 1),
                    "std_rate": round(statistics.stdev(overall_rates) if len(overall_rates) > 1 else 0, 1),
                    "min_rate": round(min(overall_rates), 1),
                    "max_rate": round(max(overall_rates), 1)
                }
            }
            
            print(f"  클러스터 {cluster_id}: 평균 {cluster_stats[str(cluster_id)]['overall_stats']['avg_rate']}%, "
                  f"표준편차 {cluster_stats[str(cluster_id)]['overall_stats']['std_rate']}%")
        
        # 4. 결과 저장
        self.save_cluster_stats(cluster_stats, period_id)
        print(f"클러스터 통계 계산 완료: {len(cluster_stats)}개 클러스터")
        
        return cluster_stats
    
    def get_team_cluster_info(self, team_id: int, period_id: int) -> Optional[Dict]:
        """특정 팀의 클러스터 정보 조회"""
        cluster_stats = self.load_cluster_stats(period_id)
        
        for cluster_id, stats in cluster_stats.items():
            if team_id in stats["teams"]:
                similar_teams = [tid for tid in stats["teams"] if tid != team_id]
                return {
                    "cluster_id": int(cluster_id),
                    "similar_teams": similar_teams,
                    "cluster_stats": stats["overall_stats"],
                    "reliability": self.get_comparison_reliability(stats["team_count"])
                }
        
        return None
    
    def get_comparison_reliability(self, cluster_teams_count: int) -> str:
        """비교 신뢰도 평가"""
        if cluster_teams_count >= 4:
            return "높음"
        elif cluster_teams_count >= 3:
            return "보통"
        elif cluster_teams_count >= 2:
            return "낮음"
        else:
            return "불가"


class TeamPerformanceComparator:
    """팀 성과 비교 통합 클래스"""
    
    def __init__(self, cache_dir="./data/cache"):
        self.stats_manager = ClusterPerformanceStatsManager(cache_dir)
        self.clustering_analyzer = TeamClusteringAnalyzer()
    
    def analyze_team_cluster_performance(self, team_id: int, period_id: int, force_recalculate: bool = False) -> Dict:
        """팀 클러스터 성과 분석 실행"""
        print(f"=== 팀 성과 비교 분석 시작: 팀 {team_id} (Q{period_id}) ===")
        
        try:
            # 1. 클러스터 통계 계산/로드
            cluster_stats = self.stats_manager.calculate_cluster_performance_stats(period_id, force_recalculate)
            
            # 2. 우리팀 클러스터 정보 조회
            team_cluster_info = self.stats_manager.get_team_cluster_info(team_id, period_id)
            
            if not team_cluster_info:
                return {
                    "success": False,
                    "error": f"팀 {team_id}의 클러스터 정보를 찾을 수 없습니다.",
                    "team_cluster_info": None
                }
            
            print(f"팀 {team_id} → 클러스터 {team_cluster_info['cluster_id']}")
            print(f"유사팀 {len(team_cluster_info['similar_teams'])}개: {team_cluster_info['similar_teams']}")
            print(f"비교 신뢰도: {team_cluster_info['reliability']}")
            
            return {
                "success": True,
                "team_cluster_info": team_cluster_info,
                "cluster_stats": cluster_stats
            }
            
        except Exception as e:
            print(f"팀 성과 비교 분석 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "team_cluster_info": None
            }
    
    def get_cluster_status(self, period_id: int) -> Dict:
        """클러스터 상태 조회"""
        cache_file = self.stats_manager.get_cache_file_path(period_id)
        
        status = {
            "cache_file_exists": os.path.exists(cache_file),
            "period_id": period_id,
            "cache_file_path": cache_file
        }
        
        if status["cache_file_exists"]:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metadata = data.get("metadata", {})
                    cluster_stats = data.get("cluster_stats", {})
                    
                    status.update({
                        "total_clusters": metadata.get("total_clusters", 0),
                        "total_teams": metadata.get("total_teams", 0),
                        "created_at": metadata.get("created_at", "Unknown"),
                        "cluster_distribution": {
                            cluster_id: len(stats["teams"])
                            for cluster_id, stats in cluster_stats.items()
                        }
                    })
            except Exception as e:
                status["error"] = f"캐시 파일 읽기 실패: {e}"
        
        return status


# 모듈 실행 시 기본 설정
if __name__ == "__main__":
    print("team_performance_comparator.py는 모듈로 사용됩니다.")
    print("실행하려면 run_team_performance_analysis.py를 사용하세요.")