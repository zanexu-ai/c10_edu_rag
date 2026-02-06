# 1.导包

"""
#查看所有数据
show databases;
#创建数据库
create database if not exists  subjects_kg;
#切库
use subjects_kg;
#查看当前库下所有的表
show tables;

"""
import logging

import pymysql  # 简建立mysql数据库连接 执行sql数据(增删查)
import pandas as pd  # 读取csv文件 ,批量插入数据到mysql
import sys, os  # sys 管理系统路径   os:处理文件路径

from base.config import conf

logger = logging.getLogger()


# print(f'MYSQL_HOST: {conf.MYSQL_HOST}')
# print(f'MYSQL_USER: {conf.MYSQL_USER}')
# print(f'MYSQL_PASSWORD: {conf.MYSQL_PASSWORD}')
# print(f'MYSQL_DATABASE: {conf.MYSQL_DATABASE}')


# todo 1 定义mysql客户端类(封装mysql连接,表操作,数据增查,关闭连接等功能)
class MySQLClient:
    def __init__(self):
        # 1.设置日志记录器:方便各个方法记录日志
        self.logger = logger

        # 2.实例化配置文件,完成数据库配置文件获取

        try:
            self.connection = pymysql.connect(
                host=conf.MYSQL_HOST,
                port=int(conf.MYSQL_PORT),
                user=conf.MYSQL_USER,
                password=conf.MYSQL_PASSWORD,
                database=conf.MYSQL_DATABASE
            )
            # 3.创建游标对象(用于执行sql语句,操作数据库,数据表等)
            self.cursor = self.connection.cursor()
            # 4.记录INFO级别日志:确定mysql连接成功
            # self.logger.info("MySQL数据库连接成功")
        except Exception as e:
            self.logger.error(f"MySQL数据库连接失败: {e}")
            # 重新抛出异常,不遮蔽错误,让调用方处理连接失败(谁调用连接,谁去可能去处理连接失败异常)
            raise

    def create_table(self):
        # 1.定义创建表的sql语句
        create_table_sql = """
            create table if not exists jpkb
            (
                id int primary key auto_increment,
                subject_name varchar(20),
                question varchar(5000),
                answer varchar(5000)
            );
        """
        # 2.建表
        try:
            self.cursor.execute(create_table_sql)  # 建表
            self.connection.commit()  # 提交
            self.logger.info("创建表成功!")

        except Exception as e:
            self.logger.error(f"创建表失败:{e}")

    def insert_data(self, csv_path):
        try:
            # 1.用pandas读取csv文件
            data = pd.read_csv(csv_path)  # 467行 3列
            # print(data)
            # 2.循环遍历csv每一行数据,执行插入操作
            for _, row in data.iterrows():
                # 2.1定义插入语句 使用占位符  避免sql注入风险
                insert_query = "insert into jpkb values (null,%s,%s,%s)"
                # 2.2执行sql语句 将csv行数据插入数据库
                self.cursor.execute(insert_query, (row['学科名称'], row['问题'], row['答案']))

            # 3.提交事务,确保所有的数据插入
            self.connection.commit()
            # 4.执行输出日志
            self.logger.info("MySQL数据插入成功")
        except Exception as e:
            self.logger.error(f"MySQL数据插入失败:{e}")
            self.connection.rollback()  # 事务回滚
            raise  # 重新抛出异常

    # 从jpkb表中获取所有的问题
    def fetch_questions(self):
        try:
            # 1.执行sql查询语句
            self.cursor.execute("select question from jpkb")

            # 2.获取查询结果
            result = self.cursor.fetchall()
            return result

        except Exception as e:
            self.logger.error(f"查询问题失败:{e}")
            return []

    # answer 回答

    def fetch_answer(self, question):
        # 获取指定问题的答案
        try:
            # 执行查询
            self.cursor.execute("SELECT answer FROM jpkb WHERE question=%s", (question,))
            # 获取结果
            tmp = self.cursor.fetchone()
            result = tmp[0] if tmp else None
            # logger.info(f'fetch_answer result:{result}')
            # 返回答案或 None
            return result
        except pymysql.MySQLError as e:
            # 记录答案获取失败
            self.logger.error(f"答案获取失败: {e}")
            # 返回 None
            return None

    # 释放资源
    def close(self):
        try:
            # 1.关闭数据库连接
            self.connection.close()
            # 2.输出日志,释放资源成功
            self.logger.info("MySQL连接已经关闭")
        except Exception as e:
            self.logger.error(f"关闭数据库连接失败:{e}")

# 1.创建mysqlclient对象
mysql_client = MySQLClient()

if __name__ == '__main__':

    # 2.建表
    # mysql_client.create_table()
    #
    # # 3.测试插入数据
    # mysql_client.insert_data(csv_path="../data/JP学科知识问答.csv")

    # 4.查询所有的问题
    q = mysql_client.fetch_questions()
    print(f'q: {q}')
    print(f'len(q): {len(q)}')

    # 5.根据问题查询答案
    a = mysql_client.fetch_answer("lxml的tree报错")
    print(f'a: {a}')
