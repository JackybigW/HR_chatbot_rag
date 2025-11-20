import streamlit as st
from rag_pipeline import LLM_using_RAG, reprompt_agent, retrieve


st.set_page_config(page_title="HR问答助手", page_icon="💼")
st.title("AFY专属人力资源问答助手")
st.write("欢迎使用AFY专属人力资源问答助手，年假，调休，离职，入职，等等问题都可以问！")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("快来试试吧！")

use_rag = st.checkbox("使用员工手册回答问题", value = True)

if user_input:
    if use_rag:
        answer = LLM_using_RAG(user_input, model = "gpt-5-nano")
    else:
        from openai import OpenAI
        import os
        api_key = os.environ['ZZZ_API_KEY']
        client = OpenAI(api_key=api_key, base_url="https://api.zhizengzeng.com/v1")
        system_prompt = "你是企业HR助手，回答员工关于公司规则的问题。"
        user_prompt = f"用户问题: {user_input}\n请根据你的知识回答，不使用任何参考资料。"
        response = client.chat.completions.create(
            messages=[
                {"role":"system", "content":system_prompt},
                {"role":"user", "content":user_prompt}
            ],
            model="gpt-5-nano"
        )
        answer = response.choices[0].message.content
    st.session_state.chat_history.append({"user": user_input, "bot": answer})

# Display chat history
for chat in st.session_state.chat_history:
    st.markdown(f"**你:** {chat['user']}")
    st.markdown(f"**助手:** {chat['bot']}")

