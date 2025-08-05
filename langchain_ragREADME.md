<img width="1240" height="743" alt="fdeb1f5bd1eb4b0523e67c3712e6192" src="https://github.com/user-attachments/assets/8cabd36c-3bb8-4c21-be14-82f0482f91c3" />
基于此，我们能够实现：

* **LangChain 的多模块能力**（向量搜索 + Agent工具）
* **Streamlit 前端交互**
* **FAISS 向量数据库**
* **DashScope Embedding + DeepSeek 模型接入**
* 并完成了完整的 RAG（检索增强生成）流程

以下是各部分功能实现代码讲解：

#### 🔧 1. 导入库 & 环境初始化

```python
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
...
load_dotenv(override=True)
```

* `Streamlit` 用于构建网页界面。
* `PyPDF2` 用来读取 PDF 文本。
* `load_dotenv()` 加载 `.env` 中的 API Key，例如：

  ```dotenv
  DEEPSEEK_API_KEY=sk-xxx
  DASHSCOPE_API_KEY=xxx
  ```

---

#### 🔐 2. 加载 API 密钥与设置环境变量

```python
DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
dashscope_api_key = os.getenv("dashscope_api_key")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
```

* 从环境变量中读取 DashScope 和 DeepSeek API。
* 设置 `KMP_DUPLICATE_LIB_OK` 避免某些 MKL 多线程报错。

---

#### 🧠 3. 初始化向量 Embedding 模型

```python
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1", dashscope_api_key=dashscope_api_key
)
```

* 用阿里云 DashScope 提供的 `text-embedding-v1` 将文本转为向量表示，用于相似度搜索。

---

#### 📄 4. 处理 PDF 文本与向量化逻辑

```python
def pdf_read(pdf_doc):
    ...
def get_chunks(text):
    ...
def vector_store(text_chunks):
    ...
```

* `pdf_read`：逐页读取 PDF 内容并拼接。
* `get_chunks`：将长文本切片为多个段落（chunk），每段 1000 字，重叠 200 字。
* `vector_store`：用 FAISS 建立向量索引，并保存到本地 `faiss_db/`。

---

#### 🔁 5. Agent对话链 + 工具调用（核心 RAG）

```python
def get_conversational_chain(tools, ques):
    llm = init_chat_model("deepseek-chat", model_provider="deepseek")
    ...
    agent_executor = AgentExecutor(...)
    response = agent_executor.invoke({"input": ques})
    ...
```

* 初始化 DeepSeek 模型为 Agent。
* 使用 LangChain 的 `create_tool_calling_agent` 构造 Agent，输入：

  * prompt（你设定的系统角色）
  * 工具（retriever 工具）
* `AgentExecutor.invoke`：LangChain 自动判断是否调用工具，完成“读取上下文 → 查询 → 回答”流程。

---

#### 🔍 6. 用户提问逻辑（调用 FAISS）

```python
def user_input(user_question):
    ...
    new_db = FAISS.load_local("faiss_db", embeddings, ...)
    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", ...)
    get_conversational_chain(retrieval_chain, user_question)
```

* 加载本地 FAISS 向量库；
* 将其转为 LangChain 的检索工具；
* 交由 Agent 调用完成回答。

---

#### 🧠 7. 检查数据库是否存在

```python
def check_database_exists():
    return os.path.exists("faiss_db") and os.path.exists("faiss_db/index.faiss")
```

简单检查本地是否已有向量化数据。

---

#### 🌐 8. 主界面逻辑（Streamlit）

```python
def main():
    st.set_page_config(...)
    ...
```

* 页面标题与界面配置。
* `st.columns` 分栏：左边显示提示，右边放置“清空数据库”按钮。
* 主输入框：`st.text_input("请输入问题")`

  * 只有当数据库存在时才能提问。
* 侧边栏：

  * PDF 上传器；
  * 提交按钮（处理上传的 PDF → 分片 → 向量化 → 存储）。

---

#### 🎯 9. 提交 PDF 后执行的逻辑

```python
if process_button:
    raw_text = pdf_read(pdf_doc)
    ...
    text_chunks = get_chunks(raw_text)
    vector_store(text_chunks)
```

* 当点击“提交并处理”后：

  1. 读取上传的 PDF；
  2. 切片文本；
  3. 向量化入库；
  4. 弹出气球提示，并 `st.rerun()` 刷新页面状态。

---

#### 📎 项目结构总结

| 模块                | 说明                             |
| ----------------- | ------------------------------ |
| 🧾 PDF解析          | 读取用户上传的 PDF                    |
| ✂️ 文本切片           | 按段落分割内容                        |
| 📊 向量化            | DashScope Embedding + FAISS 建库 |
| 🔁 查询接口           | 用户输入 → 召回相关 chunk              |
| 🤖 DeepSeek Agent | 调用检索工具并给出回答                    |
| 💻 UI层            | Streamlit 实现全部交互               |
&emsp;&emsp;其中LangChain RAG核心功能相关代码如下：
- Step 1：PDF 文件上传与文本提取

&emsp;&emsp;使用 `st.file_uploader()` 组件支持多文件上传，并通过 `PyPDF2.PdfReader` 对每页内容进行提取，组合为整体文本。

```python
    def pdf_read(pdf_doc):
        text = ""
        for pdf in pdf_doc:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
```
- Step 2：文本分块与向量数据库构建

&emsp;&emsp;使用 `RecursiveCharacterTextSplitter` 将长文档切割为固定长度（1000字）+ 重叠（200字）的小块，将文本块通过 `DashScopeEmbeddings` 嵌入为向量，使用 `FAISS` 本地存储向量数据库。


```python
    chunks = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")
```

- Step 3：用户提问与语义检索

&emsp;&emsp;通过 `Streamlit` 获取用户输入问题，如果向量数据库存在，则加载 `FAISS` 检索器，使用 `create_retriever_tool()` 构建 `LangChain` 工具，交由 `AgentExecutor` 执行，自动调用检索器并生成答案。

```python
    retrieval_chain = create_retriever_tool(retriever, ...)
    agent = create_tool_calling_agent(llm, [retrieval_chain], prompt)
    response = agent_executor.invoke({"input": ques})
```
&emsp;&emsp;
