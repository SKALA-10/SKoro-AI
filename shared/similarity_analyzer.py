# similarity_analyzer.py
# SimilarityAnalyzer - 팀 + 개인 유사도 분석 구현

import pandas as pd
import numpy as np
import json
import re
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

db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


class SimilarityDB:
    """유사도 분석 전용 DB 클래스"""
    
    def __init__(self):
        self.engine = engine
    
    def fetch_all_team_kpis(self) -> List[Dict]:
        """전사 모든 팀의 KPI 데이터 조회"""
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
    
    def fetch_individual_data_by_period(self, period_id: int) -> List[Dict]:
        """특정 분기의 개인 task 데이터 조회 (분기별 누적 task_summary 포함)"""
        with self.engine.connect() as connection:
            query = text("""
                SELECT 
                    e.emp_no,
                    e.emp_name,
                    e.cl,
                    e.position,
                    t.task_id,
                    t.task_name,
                    t.target_level,
                    ts.task_summary,
                    te.team_name,
                    h.headquarter_name,
                    h.headquarter_id
                FROM employees e
                JOIN tasks t ON e.emp_no = t.emp_no
                JOIN task_summaries ts ON t.task_id = ts.task_id
                JOIN teams te ON e.team_id = te.team_id
                JOIN headquarters h ON te.headquarter_id = h.headquarter_id
                WHERE ts.period_id <= :period_id
                AND e.role != 'MANAGER'
                ORDER BY e.emp_no, t.task_id, ts.period_id
            """)
            
            results = connection.execute(query, {"period_id": period_id}).fetchall()
            return [dict(row._mapping) for row in results]


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


class TeamSimilarityAnalyzer:
    """팀 유사도 분석 클래스"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.db = SimilarityDB()
        self.team_data = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cluster_labels = None
        self.similarity_matrix = None
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
                    'headquarter_name': row['headquarter_name'],
                    'kpi_ids': []
                }
            
            team_texts[team_id].append(preprocessed_text)
            team_info[team_id]['kpi_ids'].append(row['team_kpi_id'])
        
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
    
    def vectorize_texts(self):
        """TF-IDF 벡터화 수행"""
        print("TF-IDF 벡터화 수행 중...")
        texts = [team['combined_text'] for team in self.team_data]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=50,  # 팀은 50개 특징 사용
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 1)
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        print(f"TF-IDF 매트릭스 크기: {self.tfidf_matrix.shape}")
        return self.tfidf_matrix
    
    def find_optimal_clusters(self, max_clusters=10):
        """Silhouette Score로 최적 클러스터 개수 찾기"""
        print("최적 클러스터 개수 탐색 중...")
        n_teams = len(self.team_data)
        max_clusters = min(max_clusters, n_teams - 1)
        
        if max_clusters < 2:
            print("팀 수가 너무 적어 클러스터링을 수행할 수 없습니다.")
            return 1
        
        best_score = -1
        best_k = 2
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.tfidf_matrix)
            score = silhouette_score(self.tfidf_matrix, labels)
            print(f"클러스터 {k}개: Silhouette Score = {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        print(f"최적 클러스터 개수: {best_k} (Score: {best_score:.3f})")
        return best_k
    
    def perform_clustering(self, n_clusters=None):
        """KMeans 클러스터링 수행"""
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()
        
        print(f"KMeans 클러스터링 수행 (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.tfidf_matrix)
        
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
    
    def calculate_similarity_matrix(self):
        """코사인 유사도 매트릭스 계산"""
        print("코사인 유사도 매트릭스 계산 중...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        return self.similarity_matrix
    
    def get_similar_teams(self, team_id: int, include_scores=False) -> List:
        """특정 팀의 유사 팀들 반환"""
        # 해당 팀 찾기
        team_idx = None
        for i, team in enumerate(self.team_data):
            if team['team_id'] == team_id:
                team_idx = i
                break
        
        if team_idx is None:
            print(f"팀 ID {team_id}를 찾을 수 없습니다.")
            return []
        
        target_cluster = self.team_data[team_idx]['cluster']
        similar_teams = []
        
        # 같은 클러스터 + 유사도 임계값 이상인 팀들 찾기
        for i, team in enumerate(self.team_data):
            if i == team_idx:
                continue
                
            if team['cluster'] == target_cluster:
                similarity_score = self.similarity_matrix[team_idx][i]
                
                if similarity_score >= self.similarity_threshold:
                    if include_scores:
                        similar_teams.append({
                            'team_id': team['team_id'],
                            'team_name': team['team_name'],
                            'headquarter_name': team['headquarter_name'],
                            'similarity_score': round(similarity_score, 3)
                        })
                    else:
                        similar_teams.append(team['team_id'])
        
        if include_scores:
            similar_teams.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_teams
    
    def analyze_clusters(self):
        """클러스터 분석 결과 출력"""
        print("\n=== 클러스터 분석 결과 ===")
        clusters = {}
        for team in self.team_data:
            cluster_id = team['cluster']
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(team)
        
        for cluster_id, teams in sorted(clusters.items()):
            print(f"\n클러스터 {cluster_id} ({len(teams)}개 팀):")
            for team in teams:
                print(f"  - {team['team_name']} ({team['headquarter_name']}) [팀ID: {team['team_id']}]")


class IndividualSimilarityAnalyzer:
    """개인 유사도 분석 클래스"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.db = SimilarityDB()
        self.individual_data = None
        self.grouped_data = {}  # CL별 그룹
        self.cluster_results = {}
        self.similarity_threshold = 0.1
       
    def load_individual_data(self, period_id: int):
        """개인 task 데이터 로드 및 전처리"""
        print(f"개인 Task 데이터 로드 중... (분기 {period_id})")
        raw_data = self.db.fetch_individual_data_by_period(period_id)
        
        if not raw_data:
            raise ValueError("개인 Task 데이터가 없습니다.")
        
        # 개인별로 Task 텍스트 결합 (분기별 누적 task_summary 포함)
        individual_texts = {}
        individual_info = {}
        
        for row in raw_data:
            emp_no = row['emp_no']
            task_text = f"{row['task_name']} {row['target_level']} {row['task_summary']}"
            preprocessed_text = self.preprocessor.preprocess(task_text)
            
            if emp_no not in individual_texts:
                individual_texts[emp_no] = []
                individual_info[emp_no] = {
                    'emp_name': row['emp_name'],
                    'cl': row['cl'],
                    'position': row['position'],
                    'team_name': row['team_name'],
                    'headquarter_name': row['headquarter_name'],
                    'headquarter_id': row['headquarter_id'],
                    'task_ids': []
                }
            
            individual_texts[emp_no].append(preprocessed_text)
            individual_info[emp_no]['task_ids'].append(row['task_id'])
        
        # 개인별 텍스트 결합
        self.individual_data = []
        for emp_no, texts in individual_texts.items():
            combined_text = ' '.join(texts)
            self.individual_data.append({
                'emp_no': emp_no,
                'combined_text': combined_text,
                'emp_name': individual_info[emp_no]['emp_name'],
                'cl': individual_info[emp_no]['cl'],
                'position': individual_info[emp_no]['position'],
                'team_name': individual_info[emp_no]['team_name'],
                'headquarter_name': individual_info[emp_no]['headquarter_name'],
                'headquarter_id': individual_info[emp_no]['headquarter_id'],
                'task_count': len(texts)
            })
        
        print(f"총 {len(self.individual_data)}명 개인 데이터 로드 완료")
        return self.individual_data
    
    def group_by_cl_only(self):
        """CL별로만 그룹핑 (부문 구분 제거)"""
        print("CL별 그룹핑 중...")
        
        self.grouped_data = {}
        
        for individual in self.individual_data:
            group_key = f"CL{individual['cl']}"
            
            if group_key not in self.grouped_data:
                self.grouped_data[group_key] = []
            
            self.grouped_data[group_key].append(individual)
        
        # 그룹별 개수 출력
        print("CL별 그룹 분포:")
        for group_key, individuals in self.grouped_data.items():
            print(f"  {group_key}: {len(individuals)}명")
        
        return self.grouped_data

    def cluster_by_group(self):
        """각 그룹별로 독립 클러스터링 수행"""
        print("그룹별 클러스터링 수행 중...")
        
        self.cluster_results = {}
        
        for group_key, individuals in self.grouped_data.items():
            print(f"\n=== {group_key} 클러스터링 ===")
            
            if len(individuals) < 2:
                print(f"  {group_key}: 인원이 부족하여 클러스터링 생략 (1명)")
                # 1명인 경우 클러스터 0으로 설정
                individuals[0]['cluster'] = 0
                self.cluster_results[group_key] = {
                    'individuals': individuals,
                    'cluster_labels': [0],
                    'n_clusters': 1,
                    'tfidf_matrix': None,
                    'similarity_matrix': None
                }
                continue
            
            # 텍스트 추출
            texts = [ind['combined_text'] for ind in individuals]
            
            # TF-IDF 벡터화 (개인은 30개 특징 사용)
            tfidf_vectorizer = TfidfVectorizer(
                max_features=min(30, len(texts) * 10),  # 그룹 크기에 따라 조정
                min_df=1,
                max_df=1.0,
                ngram_range=(1, 1)
            )
            
            try:
                tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
                print(f"  TF-IDF 매트릭스 크기: {tfidf_matrix.shape}")
                
                # 최적 클러스터 개수 찾기
                n_individuals = len(individuals)
                max_clusters = min(5, n_individuals - 1)
                
                best_score = -1
                best_k = 1
                
                if max_clusters >= 2:
                    for k in range(2, max_clusters + 1):
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(tfidf_matrix)
                        score = silhouette_score(tfidf_matrix, labels)
                        print(f"    클러스터 {k}개: Silhouette Score = {score:.3f}")
                        
                        if score > best_score:
                            best_score = score
                            best_k = k
                    
                    print(f"  최적 클러스터 개수: {best_k} (Score: {best_score:.3f})")
                else:
                    best_k = 1
                    print(f"  인원이 적어 클러스터링 생략: {n_individuals}명")
                
                # 최종 클러스터링
                if best_k > 1:
                    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(tfidf_matrix)
                else:
                    cluster_labels = [0] * len(individuals)
                
                # 클러스터 결과를 개인 데이터에 저장
                for i, individual in enumerate(individuals):
                    individual['cluster'] = int(cluster_labels[i])
                
                # 코사인 유사도 계산
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # 결과 저장
                self.cluster_results[group_key] = {
                    'individuals': individuals,
                    'cluster_labels': cluster_labels,
                    'n_clusters': best_k,
                    'tfidf_matrix': tfidf_matrix,
                    'similarity_matrix': similarity_matrix,
                    'silhouette_score': best_score if best_k > 1 else 0.0
                }
                
                # 클러스터별 분포 출력
                cluster_counts = {}
                for label in cluster_labels:
                    cluster_counts[label] = cluster_counts.get(label, 0) + 1
                
                print(f"  클러스터별 분포:")
                for cluster_id, count in sorted(cluster_counts.items()):
                    print(f"    클러스터 {cluster_id}: {count}명")
                    
            except Exception as e:
                print(f"  {group_key} 클러스터링 실패: {e}")
                # 실패한 경우 모두 클러스터 0으로 설정
                for individual in individuals:
                    individual['cluster'] = 0
                
                self.cluster_results[group_key] = {
                    'individuals': individuals,
                    'cluster_labels': [0] * len(individuals),
                    'n_clusters': 1,
                    'tfidf_matrix': None,
                    'similarity_matrix': None
                }
        
        return self.cluster_results

    def get_similar_individuals(self, emp_no: str, include_scores=False) -> List:
        """특정 개인의 유사 개인들 반환 (같은 클러스터, 다른 팀, 임계값 이상)"""
        # 해당 개인 찾기
        target_individual = None
        target_group = None
        target_idx = None
        
        for group_key, individuals in self.grouped_data.items():
            for i, individual in enumerate(individuals):
                if individual['emp_no'] == emp_no:
                    target_individual = individual
                    target_group = group_key
                    target_idx = i
                    break
            if target_individual:
                break
        
        if not target_individual:
            print(f"직원 {emp_no}를 찾을 수 없습니다.")
            return []
        
        # 유사도 매트릭스 가져오기
        group_result = self.cluster_results[target_group]
        similarity_matrix = group_result['similarity_matrix']
        
        if similarity_matrix is None:
            # 유사도 매트릭스가 없는 경우 (인원 부족 등) 클러스터만 사용
            target_cluster = target_individual['cluster']
            target_team = target_individual['team_name']
            similar_individuals = []
            
            for individual in self.grouped_data[target_group]:
                if (individual['emp_no'] != emp_no and 
                    individual['cluster'] == target_cluster and
                    individual['team_name'] != target_team):
                    similar_individuals.append(individual['emp_no'])
            
            return similar_individuals
        
        # 임계값을 사용한 유사도 검사
        target_cluster = target_individual['cluster']
        target_team = target_individual['team_name']
        similar_individuals = []
        
        for i, individual in enumerate(self.grouped_data[target_group]):
            if (individual['emp_no'] != emp_no and 
                individual['team_name'] != target_team):  # 다른 팀만
                
                # 같은 클러스터이고 임계값 이상인 경우
                if individual['cluster'] == target_cluster:
                    similarity_score = similarity_matrix[target_idx][i]
                    
                    if similarity_score >= self.similarity_threshold:
                        if include_scores:
                            similar_individuals.append({
                                'emp_no': individual['emp_no'],
                                'emp_name': individual['emp_name'],
                                'position': individual['position'],
                                'team_name': individual['team_name'],
                                'similarity_score': round(similarity_score, 3)
                            })
                        else:
                            similar_individuals.append(individual['emp_no'])
        
        # 유사도 점수 기준으로 정렬 (높은 순)
        if include_scores and similar_individuals:
            similar_individuals.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_individuals
    
    def analyze_individual_clusters(self):
        """개인 클러스터 분석 결과 출력"""
        print("\n=== 개인 클러스터 분석 결과 ===")
        
        for group_key, result in self.cluster_results.items():
            print(f"\n{group_key}:")
            
            individuals = result['individuals']
            clusters = {}
            
            for individual in individuals:
                cluster_id = individual['cluster']
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(individual)
            
            for cluster_id, cluster_individuals in sorted(clusters.items()):
                print(f"  클러스터 {cluster_id} ({len(cluster_individuals)}명):")
                for individual in cluster_individuals:
                    print(f"    - {individual['emp_name']}({individual['emp_no']}) - {individual['position']}")


class SimilarityCache:
    """유사도 분석 결과 캐시 관리 클래스"""
    
    def __init__(self, cache_dir="./data/cache"):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "similarity_cache_2024.json")
        
        # 캐시 디렉토리 생성
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_cache(self) -> Dict:
        """캐시 파일 로드"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"캐시 파일 로드 실패: {e}")
                return self._create_empty_cache()
        else:
            return self._create_empty_cache()
    
    def _create_empty_cache(self) -> Dict:
        """빈 캐시 구조 생성"""
        return {
            "teams": {},
            "individuals": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "teams_updated_at": None,
                "individuals_last_quarter": None
            }
        }
    
    def save_team_results(self, team_analyzer: 'TeamSimilarityAnalyzer'):
        """팀 유사도 결과 저장 (연초 1회)"""
        print("팀 유사도 결과를 캐시에 저장 중...")
        
        cache = self.load_cache()
        
        # 팀 클러스터링 결과 저장
        teams_data = {}
        
        for team in team_analyzer.team_data:
            team_id = team['team_id']
            
            # 유사 팀 목록 생성
            similar_teams = team_analyzer.get_similar_teams(team_id, include_scores=True)
            
            teams_data[str(team_id)] = {
                "team_info": {
                    "team_name": team['team_name'],
                    "headquarter_name": team['headquarter_name'],
                    "cluster": team['cluster']
                },
                "similar_teams": [
                    {
                        "team_id": sim_team['team_id'],
                        "team_name": sim_team['team_name'],
                        "similarity_score": sim_team['similarity_score']
                    }
                    for sim_team in similar_teams
                ],
                "similarity_config": {
                    "threshold": team_analyzer.similarity_threshold,
                    "cluster_method": "kmeans"
                }
            }
        
        # 캐시에 저장
        cache["teams"] = teams_data
        cache["metadata"]["teams_updated_at"] = datetime.now().isoformat()
        cache["metadata"]["last_updated"] = datetime.now().isoformat()
        
        self._save_cache(cache)
        print(f"팀 유사도 결과 저장 완료: {len(teams_data)}개 팀")
    
    def save_individual_results(self, individual_analyzer: 'IndividualSimilarityAnalyzer', quarter: str):
        """개인 유사도 결과 저장 (분기별 추가)"""
        print(f"개인 유사도 결과를 캐시에 저장 중... (분기: {quarter})")
        
        cache = self.load_cache()
        
        # 개인 클러스터링 결과 저장
        individuals_data = {}
        
        for group_key, result in individual_analyzer.cluster_results.items():
            cl_data = {}
            
            # 클러스터별로 개인 목록 생성
            clusters = {}
            for individual in result['individuals']:
                cluster_id = individual['cluster']
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                
                clusters[cluster_id].append({
                    "emp_no": individual['emp_no'],
                    "emp_name": individual['emp_name'],
                    "position": individual['position'],
                    "team_name": individual['team_name'],
                    "headquarter_name": individual['headquarter_name']
                })
            
            cl_data = {
                "clusters": clusters,
                "cluster_config": {
                    "n_clusters": result['n_clusters'],
                    "silhouette_score": result.get('silhouette_score', 0.0),
                    "threshold": individual_analyzer.similarity_threshold
                },
                "total_individuals": len(result['individuals'])
            }
            
            individuals_data[group_key] = cl_data
        
        # 캐시에 분기별로 저장
        if "individuals" not in cache:
            cache["individuals"] = {}
        
        cache["individuals"][quarter] = individuals_data
        cache["metadata"]["individuals_last_quarter"] = quarter
        cache["metadata"]["last_updated"] = datetime.now().isoformat()
        
        self._save_cache(cache)
        print(f"개인 유사도 결과 저장 완료: {quarter} - {len(individuals_data)}개 CL 그룹")
    
    def _save_cache(self, cache: Dict):
        """캐시 파일 저장"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"캐시 파일 저장 실패: {e}")
    
    def get_similar_teams_from_cache(self, team_id: int) -> List[int]:
        """캐시에서 유사 팀 목록 조회"""
        cache = self.load_cache()
        
        if "teams" not in cache or str(team_id) not in cache["teams"]:
            return []
        
        team_data = cache["teams"][str(team_id)]
        return [sim_team["team_id"] for sim_team in team_data["similar_teams"]]
    
    def get_similar_individuals_from_cache(self, emp_no: str, quarter: str, cl: int) -> List[str]:
        """캐시에서 유사 개인 목록 조회"""
        cache = self.load_cache()
        
        if ("individuals" not in cache or 
            quarter not in cache["individuals"] or 
            f"CL{cl}" not in cache["individuals"][quarter]):
            return []
        
        cl_data = cache["individuals"][quarter][f"CL{cl}"]
        
        # 해당 개인이 속한 클러스터 찾기
        target_cluster = None
        target_team = None
        
        for cluster_id, individuals in cl_data["clusters"].items():
            for individual in individuals:
                if individual["emp_no"] == emp_no:
                    target_cluster = cluster_id
                    target_team = individual["team_name"]
                    break
            if target_cluster is not None:
                break
        
        if target_cluster is None:
            return []
        
        # 같은 클러스터, 다른 팀의 개인들 반환
        similar_individuals = []
        for individual in cl_data["clusters"][target_cluster]:
            if (individual["emp_no"] != emp_no and 
                individual["team_name"] != target_team):
                similar_individuals.append(individual["emp_no"])
        
        return similar_individuals
    
    def get_cache_status(self) -> Dict:
        """캐시 상태 조회"""
        cache = self.load_cache()
        
        status = {
            "cache_file_exists": os.path.exists(self.cache_file),
            "teams_cached": len(cache.get("teams", {})) > 0,
            "individuals_quarters": list(cache.get("individuals", {}).keys()),
            "metadata": cache.get("metadata", {})
        }
        
        return status


class SimilarityAnalyzer:
    """통합 유사도 분석 클래스"""
    
    def __init__(self, cache_dir="./data/cache"):
        self.team_analyzer = TeamSimilarityAnalyzer()
        self.individual_analyzer = IndividualSimilarityAnalyzer()
        self.cache = SimilarityCache(cache_dir)
    
    def analyze_teams(self, save_to_cache=True):
        """팀 유사도 분석 실행"""
        print("=== 팀 유사도 분석 시작 ===")
        try:
            self.team_analyzer.load_team_data()
            self.team_analyzer.vectorize_texts()
            self.team_analyzer.perform_clustering()
            self.team_analyzer.calculate_similarity_matrix()
            self.team_analyzer.analyze_clusters()
            
            # 캐시 저장
            if save_to_cache:
                self.cache.save_team_results(self.team_analyzer)
            
            print("팀 유사도 분석 완료!")
            return True
        except Exception as e:
            print(f"팀 유사도 분석 실패: {e}")
            return False
    
    def analyze_individuals(self, period_id: int, save_to_cache=True):
        """개인 유사도 분석 실행"""
        quarter = f"Q{period_id}"
        print(f"=== 개인 유사도 분석 시작 (분기 {quarter}) ===")
        
        try:
            self.individual_analyzer.load_individual_data(period_id)
            self.individual_analyzer.group_by_cl_only()
            self.individual_analyzer.cluster_by_group()
            self.individual_analyzer.analyze_individual_clusters()
            
            # 캐시 저장
            if save_to_cache:
                self.cache.save_individual_results(self.individual_analyzer, quarter)
            
            print("개인 유사도 분석 완료!")
            return True
        except Exception as e:
            print(f"개인 유사도 분석 실패: {e}")
            return False
    
    def get_similar_teams(self, team_id: int, use_cache=True, include_scores=False):
        """유사 팀 조회 (캐시 우선, 실패시 실시간 분석)"""
        if use_cache:
            cached_teams = self.cache.get_similar_teams_from_cache(team_id)
            if cached_teams:
                return cached_teams
        
        # 캐시에 없으면 실시간 분석 결과 사용
        if self.team_analyzer.team_data is not None:
            return self.team_analyzer.get_similar_teams(team_id, include_scores)
        else:
            print("팀 분석이 수행되지 않았습니다. analyze_teams()를 먼저 실행하세요.")
            return []
    
    def get_similar_individuals(self, emp_no: str, period_id: int, cl: int, use_cache=True, include_scores=False):
        """유사 개인 조회 (캐시 우선, 실패시 실시간 분석)"""
        if use_cache:
            quarter = f"Q{period_id}"
            cached_individuals = self.cache.get_similar_individuals_from_cache(emp_no, quarter, cl)
            if cached_individuals:
                return cached_individuals
        
        # 캐시에 없으면 실시간 분석 결과 사용
        if self.individual_analyzer.cluster_results:
            return self.individual_analyzer.get_similar_individuals(emp_no, include_scores)
        else:
            print(f"개인 분석이 수행되지 않았습니다. analyze_individuals({period_id})를 먼저 실행하세요.")
            return []
    
    def get_cache_status(self):
        """캐시 상태 조회"""
        return self.cache.get_cache_status()


# 모듈로 사용될 때를 위한 기본 설정
if __name__ == "__main__":
    print("similarity_analyzer.py는 모듈로 사용됩니다.")
    print("실행하려면 run_similarity_analysis.py를 사용하세요.")