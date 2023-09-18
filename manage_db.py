import pandas as pd
import pymysql
import json
from datetime import datetime

class NewsDB:
    """
    클래스 설명
    """

    def __init__(self, db_config:dict, cursor_type="tuple") -> None:
        """
        데이터베이스 접속
        인자 : 데이터베이스 접속정보
        """
        db_config['port'] = int(db_config.get('port', '3306'))
        self.DB = pymysql.connect(**db_config)

        # 데이터베이스에서 select 해서 가져와도 상관없음.        
        try:
            self.category1_info = json.load(open('./category1.json', 'r'))
            self.category2_info = json.load(open('./category2.json', 'r'))
            self.category2_info = {tuple(value): idx for idx, value in self.category2_info.items()}

            self.PLATFORM_DICT = {
                '다음': 1,
                '네이버': 2
            }

        except:
            print('카테고리 정보가 없습니다.')

        if cursor_type == 'dict':
            self.cursor_type = pymysql.cursors.DictCursor
        else:
            self.cursor_type = None
    
    def __del__(self) -> None:
        """
        데이터베이스 연결 해제
        """

        self.DB.close()


    def insert_news(self, news_df):
        """
        인자 : 뉴스기사 데이터프레임
        
        우선 데이터프레임의 column명 체크하여 News 테이블의 칼럼이름과 일치하지 않을 경우 에러 발생시키기

        insert SQL문 생성
        execute 대신 execute_many 메서드로 한번에 삽입

        1. 플랫폼 정보 id로 변환
        2. 메인카테고리 숫자로 변환
        3. 서브카테고리 숫자로 변환
        4. DB에 적재

        """
        # 기본적으로는 NOT NULL인 값들만 체크
        required_columns = ['platform', 'category1', 'category2', 'content', 'title', 'date_upload', 'writer', 'url', 'sticker']
        assert not set(required_columns) - set(news_df.columns), '테이블 칼럼이 부족합니다.'
        

        # 테이블에 삽입할 수 있는 데이터로 변환
        news_df['platform_id'] = news_df['platform'].map(self.PLATFORM_DICT)
        news_df['cat1_id'] = news_df['category1'].map(self.category1_info)
        cat2_ids = []
        for cat1, cat2 in news_df[['category1', 'category2']].iloc:
            try:
                cat2_ids.append(self.category2_info[(cat1, cat2)])
            except:
                cat2_ids.append('')
        news_df['cat2_id'] = cat2_ids
        news_df['writer'] = news_df['writer'].apply(lambda x : x[:16])

        news_df['sticker'] = news_df['sticker'].apply(json.dumps)

        # 데이터 INSERT
        target_column = ['platform_id', 'cat1_id', 'cat2_id', 'title', 'content', 'date_upload', 'writer', 'url', 'sticker']
        table = 'NEWS'
        columns = ','.join(target_column)
        values = news_df[target_column].values.tolist()

        sql = f"INSERT INTO {table}({columns}) " \
                  "VALUES ("  + ','.join(["%s"]*len(values[0])) + ");"
        
        try:
            with self.DB.cursor() as cur:
                cur.executemany(sql, values)
                self.DB.commit()
            return True
        except:
            import traceback
            traceback.print_exc()
            self.DB.rollback()
            return False

    # insert_user()
    # 댓글 데이터프레임에서 유저 정보만 뽑아서 넣는 방법도 있음
    def insert_user(self, user_df):
        """
        유저 insert
        """
        user_df = user_df.drop_duplicates(subset='user_id')

        target_column = ['platform_id', 'user_id']

        table = 'USER'
        column = ','.join(target_column)
        values = user_df[target_column].values.tolist()
        sql = f'INSERT IGNORE INTO {table}({column}) VALUES (%s, %s)'
        try:
            with self.DB.cursor() as cur:
                cur.executemany(sql, values)
                self.DB.commit()
            return True
        except:
            import traceback
            traceback.print_exc()
            self.DB.rollback()
            return False


    def change_comment_df(self, df):
        """
        인자 : 댓글 데이터프레임

        데이터프레임 칼럼 체크하여 Comment 테이블의 칼럼과 일치하지 않을 경우 에러
        
        1. 유저 테이블에서 있는지 체크, id값 있을 경우 변환
        2. 신규 유저일 경우 유저 테이블에 추가, id값 가져오기 (DB에 유저 정보가 저장되어있다면 가져오기)
        3. url을 통해 코멘트 별 뉴스기사 id 가져오기 (select)
        """
        users = self.select('*', 'user')
        user_dict = {u[2]: u[0] for u in users}
        df['user_id'] = df['user_id'].map(user_dict)

        news = self.select('url,id', 'news')
        news_dict = {n[0]: n[1] for n in news}
        df['news_id'] = df['url'].map(news_dict)


        df.dropna(inplace=True)
        df['user_id'] = df['user_id'].astype(int)
        df['news_id'] = df['news_id'].astype(int)

        return df
    
    def select(self, column_qry, table):
        """
        셀렉트 all
        """
        sql_qr = "SELECT {0} FROM {1}".format(column_qry, table)

        with self.DB.cursor() as cur:
            cur.execute(sql_qr)
            return cur.fetchall()

    def insert_comment(self, comment_df):
        """
        인자 : 댓글 데이터프레임

        데이터프레임 칼럼 체크하여 Comment 테이블의 칼럼과 일치하지 않을 경우 에러

        1. 댓글 id로 변환하는 함수 호출하여 변환한 데이터프레임 가져오기
        2. DB에 적재
        """        
        # 기본적으로는 NOT NULL인 값들만 체크
        required_columns = ['comment', 'user_id', 'user_name', 'url', 'date_upload']
        assert not set(required_columns) - set(comment_df.columns), '테이블 칼럼이 부족합니다.'

        if not self.insert_user(comment_df[['platform_id', 'user_id']]):
            print('유저 삽입 실패')
            return False
        
        # 변환
        comment_df = self.change_comment_df(comment_df)

        target_column = ['news_id', 'date_upload', 'user_id', 'comment']

        table = 'COMMENT'
        column = ','.join(target_column)
        values = comment_df[target_column].values.tolist()
        sql = f'INSERT INTO {table}({column}) VALUES (%s, %s, %s, %s)'
        try:
            with self.DB.cursor() as cur:
                cur.executemany(sql, values)
                self.DB.commit()
            return True
        except:
            import traceback
            traceback.print_exc()
            self.DB.rollback()
            return False

    # 각 인원이 ERD 통해 데이터베이스에 테이블 생성해서 수집한 데이터로 테스트해 볼 것
        

    def select_news(self, start_date=None, end_date=None, platform=None, category1=None, category2:tuple=None):
        """
        인자 : 데이터를 꺼내올 때 사용할 parameters 
        (어떻게 검색(필터)해서 뉴스기사를 가져올 것인지)

        DB에 들어있는 데이터를 꺼내올 것인데, 어떻게 꺼내올지를 고민

        인자로 받은 파라미터 별 조건을 넣은 select SQL문 작성

        SQL문에 추가할 내용들
        1. 가져올 칼럼
        2. JOIN할 경우 JOIN문 (플랫폼, 카테고리)
        3. WHERE 조건문
        4. LIMIT, OFFSET 등 처리
        """

        where_sql = []

        if start_date and end_date:
            where_sql.append(f"date_upload BETWEEN '{start_date}' AND '{end_date}'")
        elif start_date:
            where_sql.append(f"date_upload >= '{start_date}'")
        elif end_date:
            where_sql.append(f"date_upload <= '{end_date}'")

        if category1:
            cat1_id = self.category1_info[category1]
            where_sql.append(f"cat1_id={cat1_id}")

        if category2:
            # 튜플형식 (메인, 서브)
            cat2_id = self.category2_info[category2]
            where_sql.append(f"cat2_id={cat2_id}")

        if platform:
            where_sql.append(f"platform_id={self.PLATFORM_DICT[platform]}")

        main_query = f'SELECT id,platform_id,cat1_id,cat2_id,press,writer,title,content,date_upload,url FROM NEWS'

        if where_sql:
            main_query += f' WHERE {" AND ".join(where_sql)}'

        # 1GB Ram 제한 (limit, offset)
        pagination_sql = ' LIMIT 100000 OFFSET {}'
        offset = 0
        final_result = []
        while True:
            with self.DB.cursor() as cur:
                cur.execute(main_query + pagination_sql.format(offset))
                result = cur.fetchall()
                final_result.extend(result)

            if len(result) < 100000:
                break

            offset += 100000 # LIMIT

        news_column = ['news_id','platform','category1', 'category2', 'press','writer','title','content','date_upload','url']
        
        df = pd.DataFrame(final_result, columns=news_column)
        self.CATEGORY1_ID2NAME = {int(v): k for k, v in self.category1_info.items()}
        self.CATEGORY2_ID2NAME = {int(v): k for k, v in self.category2_info.items()}
        self.PLATFORM_ID2NAME = {int(v): k for k, v in self.PLATFORM_DICT.items()}

        df['platform'] = df['platform'].map(self.PLATFORM_ID2NAME)
        df['category1'] = df['category1'].map(self.CATEGORY1_ID2NAME)
        df['category2'] = df['category2'].map(self.CATEGORY2_ID2NAME)

        return df
    
    def select_user(self):
        """
        인자 : 데이터를 꺼내올 때 사용할 parameters
        (어떻게 검색(필터)해서 유저를 가져올 것인지)

        SQL문에 추가할 내용들
        1. 가져올 칼럼
        2. JOIN할 경우 JOIN문
        3. WHERE 조건문
        4. LIMIT, OFFSET 등 처리
        """

        return self.select('*', 'USER')

    
    def select_comment(self):
        """
        인자 : 데이터를 꺼내올 때 사용할 parameters
        (어떻게 검색(필터)해서 댓글을 가져올 것인지)

        SQL문에 추가할 내용들
        1. 가져올 칼럼
        2. JOIN할 경우 JOIN문 (유저정보를 같이 가져올 경우)
        3. WHERE 조건문
        4. LIMIT, OFFSET 등 처리
        """
        return self.select('*', 'COMMENT')
    

    def insert_many(self, table: str, columns: str, values: list) -> bool:
        """
        Insert Many Datas
        
        example)
        table = "Students"
        columns = "name, email, phone"
        values = [
            ('hong gildong', 'hgd123@gmail.com', '01012345678'),
            ...
        ]
        """
        sql = f"INSERT INTO {table}({columns}) " \
                  "VALUES ("  + ','.join(["%s"]*len(values[0])) + ");"
        try:
            with self.DB.cursor() as cur:
                cur.executemany(sql, values)
                self.DB.commit()
            return True
        except:
            import traceback
            traceback.print_exc()
            self.DB.rollback()
            return False



def read_config(config_path:str, splitter:str='=', encoding=None) -> dict:
    """
    config 파일 읽고 반환
    config_path = 파일 경로
    splitter = 구분 기호
    """
    temp = {}
    with open(config_path, 'r', encoding=encoding) as f:
        for l in f.readlines():
            k, v = l.rstrip().split(splitter)
            temp[k] = v
    return temp

if __name__ == '__main__':
    ### 테스트코드 작성

    db1 = read_config('./db1.config')
    # 개인 데이터베이스에 연결
    try:
        news_db = NewsDB(db1)
        print('데이터베이스 연결 성공')
    except:
        print('데이터베이스 연결 실패')

    print('플랫폼, 카테고리 삽입')
    categories = [l.strip().split(',') for l in open('./category.csv', encoding='utf-8').readlines()]
    categories[0][0] = '다음'    
    
    category_datas = []
    for category in categories:
        # 플랫폼
        if category[1].endswith('0000'):
            platform = category[0]
            category_datas.append(
                ('', '', platform, category[1])
            )
            continue
        elif category[1].endswith('00'):
            category_1 = category[0]
            category_datas.append(
                ('', category_1, platform, category[1])
            )
            continue
        else:
            category_2 = category[0]
            category_datas.append(
                (category_2, category_1, platform, category[1])
            )
    
    ### 플랫폼, 카테고리 삽입
    news_db.insert_many('PLATFORM', 'name', [('다음',), ('네이버',)])
    # (1, '다음'), (2, '네이버')
    category_1 = list(set([category[1] for category in category_datas]))
    category_1_data = [(c, )for c in category_1[1:]]
    news_db.insert_many('CATEGORY_1', 'name', category_1_data)

    category_2 = list(set([category[0] for category in category_datas]))
    # 카테고리 1 변환
    cateogry1_ids = news_db.select('*', 'CATEGORY_1')
    name_to_idx = {name: idx for idx, name in cateogry1_ids}
    idx_to_name = {idx: name for idx, name in cateogry1_ids}

    category_2_data = [(c[0], name_to_idx[c[1]]) for c in category_datas if c[0]]

    news_db.insert_many('CATEGORY_2', 'name,cat1_id', category_2_data)

    cateogry_2_ids = news_db.select('*', 'CATEGORY_2')
    category_2_info = {c[0] : (idx_to_name[c[1]], c[2]) for c in cateogry_2_ids}
    
    print('플랫폼, 카테고리 삽입완료, json으로 저장')
    json.dump(category_2_info, open('./category2.json', 'w'))
    json.dump(name_to_idx, open('./category1.json', 'w'))

    # 새로 연결
    try:
        news_db = NewsDB(db1)
    except:
        print('데이터베이스 연결 실패')

    ### insert 테스트 (뉴스, 코멘트)

    print('뉴스기사 데이터 삽입')

    ### 뉴스
    tsv_file = open('D:/Downloads/metadata.tsv', 'r', encoding='utf-8').readlines()
    datas = [l.rstrip().split('\t') for l in tsv_file]

    import pandas as pd

    df = pd.DataFrame(datas, columns=['platform', 'category1', 'category2', 'title', 'press', 'writer', 
                                    'date_upload', 'date_fix', 'url', 'content', 'sticker'])
    
    cat_dict = {'연예': '연예일반', 'e-스포츠': 'e스포츠'}
    df['category2'] = df['category2'].apply(lambda x : cat_dict[x] if x in cat_dict else x)

    from datetime import datetime
    def change_dt(x):
        if not x:
            return datetime.now()
        try:
            return datetime.fromtimestamp(int(x)/1000)
        except: 
            try:
                return datetime.strptime(x, '%Y. %m. %d. %H:%M')
            except:
                return datetime.strptime(x, '%Y-%m-%d %H:%M')
    df['date_upload'] = df['date_upload'].apply(change_dt)
    df['date_fix'] = df['date_fix'].apply(change_dt)
    
    news_db.insert_news(df)
    
    # 네이버
    tsv_file = open('D:/Downloads/enter_metadata.tsv', 'r', encoding='utf-8').readlines()
    datas = [l.rstrip().split('\t') for l in tsv_file]

    tsv_file = open('D:/Downloads/news_metadata.tsv', 'r', encoding='utf-8').readlines()
    datas += [l.rstrip().split('\t') for l in tsv_file]

    tsv_file = open('D:/Downloads/sports_metadata.tsv', 'r', encoding='utf-8').readlines()
    temp = [['네이버']+l.rstrip().split('\t') for l in tsv_file]
    datas += [t[:3] + [t[4], t[-2], t[-3], t[3], t[3], t[5], t[6], t[-1]] for t in temp]
    
    import pandas as pd

    df = pd.DataFrame(datas, columns=['platform', 'category1', 'category2', 'title', 'press', 'writer', 
                                    'date_upload', 'date_fix', 'content', 'url', 'sticker'])
    df['category2'] = df['category2'].apply(lambda x : cat_dict[x] if x in cat_dict else x)

    news_db.insert_news(df)

    # assert len(datas) == news_db.select('COUNT(*)', 'NEWS')[0][0], '뉴스 삽입 실패'

    print('뉴스기사 데이터 완료')

    ### 코멘트
    print('댓글 데이터 삽입')

    tsv_file = open('D:/Downloads/news_comments.tsv', 'r', encoding='utf-8').readlines()
    datas = [l.rstrip().split('\t') for l in tsv_file]
    datas = [d for d in datas if len(d)==6]

    comment_df = pd.DataFrame(datas, columns=['url', 'commentNo', 'user_id', 'user_name', 'comment', 'date_upload'])
    comment_df['platform_id'] = 2

    news_db.insert_comment(comment_df)

    # assert len(comment_df) == news_db.select('COUNT(*)', 'COMMENT')[0][0], '코멘트 삽입 실패'

    print('댓글 데이터 완료')

    # select 테스트 (뉴스, 코멘트, 유저)
    print('셀렉트 테스트')
    result = news_db.select_news(platform='네이버')
    print(result.head())

    result = news_db.select_comment()
    print(result[:5])

    result = news_db.select_user()
    print(result[:5])
