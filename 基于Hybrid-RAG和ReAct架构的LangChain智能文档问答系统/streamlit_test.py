
# 导入构建链的相关库

from langchain.chat_models import init_chat_model
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent,AgentExecutor
# 导入RAG系统所需的库
import os
from langchain.tools import tool
from PyPDF2 import PdfReader    #pdf读取
from langchain.text_splitter import RecursiveCharacterTextSplitter  #文档切分
from langchain_community.embeddings import DashScopeEmbeddings  #调用阿里云百炼平台的embedding模型
from langchain_community.vectorstores import FAISS  #使用FAISS向量数据库存储切分好的文本向量
# 导入记忆库
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

# 1.创建pdf读取和分块函数
## 1.1 pdf读取函数
def pdf_read(pdf_doc):
    text=""     #存储所有提取的文本内容
    for pdf in pdf_doc:     #循环读取每一个pdf文件
        print(pdf)
        pdf_reader=PdfReader(pdf)   #PdfReader读取文件后，会返回页面信息
        for page in pdf_reader.pages:   #一页一页的处理
            page_text=page.extract_text()   #提取每一页的文件内容
            text=text+page_text             #组成text
    return text

## 1.2 text切块函数
def get_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) #创建一个文本分块的实例对象
    chunks=text_splitter.split_text(text)
    return chunks

# 2.初始化向量模型
embeddings=DashScopeEmbeddings(
    model='text-embedding-v1',
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)
# 3.使用embedding模型向量化切块后的文本并存入FAISS向量数据库中——创建向量转换函数
def vector_store(chunks):
    vector=FAISS.from_texts(chunks,embedding=embeddings)
    vector.save_local("faiss_db1")
    # # 保存文本块用于BM25关键词检索
    # global text_chunks_for_bm25
    # text_chunks_for_bm25 = chunks

# 4.检查向量数据库是否存在
def check_database_exists():
    """检查FAISS数据库是否存在"""
    return os.path.exists("faiss_db1") and os.path.exists("faiss_db1/index.faiss")


# 5.使用get_chunks函数返回的chunks构建Whoosh倒排索引并存储，便于后续进行关键词搜索
from whoosh.fields import Schema, TEXT
from whoosh.index import create_in
import os
from jieba.analyse import ChineseAnalyzer
from langdetect import detect

def whoosh_index_store(chunks):
    text=" ".join(chunks)
    lang=detect(text)
    if lang=="zh":
        an=ChineseAnalyzer()
        schema=Schema(content=TEXT(stored=True,analyzer=an))
    else:
        schema=Schema(content=TEXT(stored=True))
    if not os.path.exists("whoosh_index"):
        os.mkdir("whoosh_index")
    ix=create_in("whoosh_index",schema)
    writer=ix.writer()
    for chunk in chunks:
        writer.add_document(content=chunk)
    writer.commit()


# 6.创建一个rag检索函数，用于返回根据用户问题检索到的文本内容
@tool
# def rag_search(user_inputs,path):
def rag_search(user_inputs):
    """根据用户问题从向量库中检索相关的内容"""
    print(f"正在查找工具")
    path="faiss_db"
    new_db=FAISS.load_local(path,embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_inputs,k=3)
    return "\n\n".join([doc.page_content for doc in docs])


# 6.创建一个hybrid_search函数————向量加关键词检索
    ## 6.1 导入关键词检索的库
from whoosh.index  import open_dir
from whoosh import scoring
from whoosh.qparser import QueryParser,OrGroup
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import numpy as np

def normalize(scores):
    '''对向量检索和关键词检索的分数进行归一化，缩放当统一尺度方便进行加权融合'''
    min_score=min(scores.values())
    max_score=max(scores.values())
    return {k:(v-min_score)/(max_score-min_score+1e-8) for k,v in scores.items()}

    ## 6.2 初始化rerank模型
tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
reranker=AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
    ## 6.3 权重设置
bm25_weight=0.4
faiss_weight=0.6
top_k=5

@tool
def hybrid_search(user_inputs):
    """1.向量检索——根据用户问题从向量库中检索相关的内容"""
    path="faiss_db1"
    new_db=FAISS.load_local(path,embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search_with_score(user_inputs,k=10)
    faiss_chunks = {doc.page_content: score for doc,score in docs}
    print(faiss_chunks)
    print("\n")

    """2.关键词检索——提前用户问题中的关键词，比如一些专业名词的问题"""
    ix=open_dir("whoosh_index")
    print("\n whoosh_index存在")
    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        parser=QueryParser("content",schema=ix.schema,group=OrGroup.factory(0.9))
        myquery=parser.parse(user_inputs)
        bm25_results=searcher.search(myquery,limit=10)
        print(bm25_results)
        bm25_chunks={hit["content"]:hit.score for hit in bm25_results}
        print(bm25_chunks)

    """3. 合并向量检索和关键词检索的结果并加权，对最终结果选取前top-k的结果"""
    all_chunks = {}
    if bm25_chunks:
        bm25_chunks = normalize(bm25_chunks)
    print("1\n")
    if faiss_chunks:
        faiss_chunks = normalize(faiss_chunks)

    for chunk, score in bm25_chunks.items():
        all_chunks[chunk] = all_chunks.get(chunk, 0) + bm25_weight * score
    for chunk, score in faiss_chunks.items():
        all_chunks[chunk] = all_chunks.get(chunk, 0) + faiss_weight * score
    sorted_chunks=sorted(all_chunks.items(),key=lambda x:x[1],reverse=True)
    candidate_chunks=[chunk for chunk,_ in sorted_chunks[:top_k]]

    """4.Rerank"""
    print("正在执行Rerank..")
    rerank_inputs=tokenizer(
        [f"{user_inputs}[SEP]{chunk}" for chunk in candidate_chunks],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        scores = reranker(**rerank_inputs).logits.squeeze(-1)
    reranked = sorted(zip(candidate_chunks, scores.tolist()), key=lambda x: x[1], reverse=True)

    """5. 返回最终筛选结果"""
    final_context = "\n\n".join([chunk for chunk, _ in reranked])
    return final_context
# 7.初始化模型
llm=ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)
# 8.创建对话历史
history_store={}
def get_session_history(session_id):
    if session_id not in history_store:
        history_store[session_id]=ChatMessageHistory()
    return history_store[session_id]

## 9.加载工具
# tools=[rag_search]
tools=[hybrid_search]

## 10. 构建基于ReAct的prompt
from langchain_core.prompts import PromptTemplate
template= """
    Answer the following questions as best you can. You have access to the following tools:
    {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat 3 times)
    Thought: I now know the final answer, don't need to take a action. give the Final Answer
    Final Answer: the final answer to the original input question
    Begin!
    chat_history:{chat_history}
    Question: {input}
    Thought:{agent_scratchpad}
"""
prompt = PromptTemplate.from_template(template)

# 11. 创建前端展示界面——链接多轮对话
import streamlit as st
import tempfile

def UI():
    ## 1.设置页面标题和布局
    st.set_page_config(
        page_title="智能问答系统",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    ## 2.设置界面顶部的标题和间接
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: #1f77b4; margin: 0;">🤖 智能问答系统</h1>
        <p style="color: #666; margin: 5px 0 0 0;">基于RAG技术的智能文档问答系统</p>
    </div>
    """, unsafe_allow_html=True)
    ## 3.创建页面布局
    col1,col2=st.columns([1,2])

    ## 4.设置左右两列主要的页面内容
        ### 左侧——上传文档和使用说明
    with col1:
        #### 使用说明区域
        st.markdown("### ⚙️ 使用说明")
        st.markdown("""
        1.在左侧上传PDF文档
        2.等待文档处理完成
        3.在右侧输入问题开始对话            
        """)

        #### 文档加载区域
        st.markdown("### 📁 文档上传")
        #1.文件上传组件
        uploaded_files=st.file_uploader(
            "上传PDF文档",
            type=['pdf'],
            accept_multiple_files=True,
            help="支持上传多个PDF文件"
        )
        #2.如果文档上传好了，则开始“处理文档”，点击处理文档的按钮，则会出现一个状态“向量数据库已就绪，反之则提示请先上传文档”
        if uploaded_files:
            if st.button("🚀 处理文档",use_container_width=True):
                with st.spinner("正在处理文档...."):
                    try:
                        # 保存上传的所有文件的本地路径，便于后续读取
                        temp_files=[]   #里面存储的是上传的所有pdf的本地路径

                        for file in uploaded_files:
                            temp_file=tempfile.NamedTemporaryFile(delete=False,suffix='.pdf')
                            temp_file.write(file.getvalue())
                            temp_files.append(temp_file.name)
                        
                        # 处理文档
                        print(f"加载的文件有：{temp_files}")
                        text = pdf_read(temp_files)
                        print("文档内容读取成功")
                        text_chunks = get_chunks(text)
                        print("分块成功")
                        vector_store(text_chunks)
                        print("向量库准备成功")
                        whoosh_index_store(text_chunks)
                        print("索引库准备成功")

                        #判断是否处理成功，成功，贼显示“数据库已就绪”，反之“请先上传文件或数据读取失败”
                        if check_database_exists():
                            st.success("✅ 数据库已就绪")
                            st.success(f"✅ 成功处理 {len(uploaded_files)} 个文档，共 {len(text_chunks)} 个文本块")
                        else:
                            st.warning("⚠️ 请先上传文档或数据读取失败")
                        
                    except Exception as e:
                        st.error(f"❌ 处理文档时出错: {str(e)}")
        #### 文档加载区域
        st.markdown("### 🔄 对话重置")
        if st.button("🔄 清空聊天记录"):
            st.session_state.messages=[]
            history_store["user-123"]=ChatMessageHistory()
            st.success("对话已重置，可以重新开始")
            
     ### 右侧——对话区
    with col2:
        #### 初始化对话历史
        if "messages" not in st.session_state:
            st.session_state.messages=[]

        if "conversation_chain" not in st.session_state:
            agent = create_react_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                early_stopping_method="force",
            )
            st.session_state.conversation_chain = RunnableWithMessageHistory(
                agent_executor,
                get_session_history=get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
        #### 设置对话区的大小
        chat_container = st.container(height=600)
            #### 将对话内容加载到对话区
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        #### 用户问题输入
        user_inputs=st.chat_input("请输入您的问题...",key="user_inputs")  #用户输入
        if user_inputs:
            st.session_state.messages.append({"role":"user","content":user_inputs})
            #在对话区显示用户问题
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_inputs)

            with st.spinner("正在思考...."):
                try:

                    response=st.session_state.conversation_chain.invoke({'input':user_inputs},config={"configurable": {"session_id": "user-123"}})
                    print(response.get("output","抱歉，我无法回答这个问题。"))
                    #将模型生成的内容添加到对话历史状态并显示
                    response_text=response.get("output","抱歉，我无法回答这个问题")
                    st.session_state.messages.append({"role":"assistant","content":response_text})
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.markdown(response_text)


                except Exception as e:
                    error_msg=f"生成回答时出错: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.markdown(error_msg)

if __name__=="__main__":
    UI()


    

