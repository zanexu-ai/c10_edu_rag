from langchain.prompts import PromptTemplate


# 定义RAGPrompts类, 集中存放所有关于rag的模版
class RAGPrompts:

    @staticmethod
    def rag_prompt():
        """
        创建直接检索的回答模板
        核心逻辑:
            优先基于传入的context生成答案,无上下文则用自身支持.无法回答,返回客服信息
        :param self:
        :return:
        """
        return PromptTemplate(template="""
            你是一个智能助手,帮助用户回答问题
                1.如果提供上下文,请基于上下文回答;
                2.如果没有上下文,请直接根据你的知识进行回答;
                3.如果答案来源于检索的文档,请生成结果的(回答)中说明
                上下文:{context},
                问题:{question}
                如果无法回答,请提示"信息不足,无法回答,请与人工客服联系:电话{phone}"
        """, input_variables=["context", "question", "phone"])

    @staticmethod
    def hyde_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
                假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
                问题: {query}  
                假设答案:  
                """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义子查询生成的 Prompt 模板
    @staticmethod
    def subquery_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
                将以下复杂查询分解为多个简单子查询，每行一个子查询：  
                查询: {query}  
                子查询:  
                """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义回溯问题生成的 Prompt 模板
    @staticmethod
    def backtracking_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
                将以下复杂查询简化为一个更简单的问题：  
                查询: {query}  
                简化问题:  
                """,
            #   定义输入变量
            input_variables=["query"],
        )


if __name__ == '__main__':
    rag_prompt = RAGPrompts.rag_prompt()
    result = rag_prompt.format(
        context="黑马程序员是传智教育（A股代码：003032）旗下高端IT教育品牌，成立于2012年，总部位于北京市昌平区建材城西路金燕龙办公楼一层。",
        question='机构名是什么?', phone="12345678")
    print(result)
