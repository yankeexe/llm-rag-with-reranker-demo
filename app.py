from rag import *
import streamlit as st

if __name__ == "__main__":
    # Document Upload Area
    with st.sidebar:
        st.set_page_config(page_title="RAG Question Answer")
        uploaded_file = st.file_uploader(
            "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
        )

        process = st.button(
            "⚡️ Process",
        )
        if uploaded_file and process:
            splitAndStore(uploaded_file)
            st.success("Data added to the vector store!")

    # Question and Answer Area
    st.header("🗣️ RAG Question Answer")
    # resetSp=st.button("reset")
    # if resetSp:
    #   st.write('reseted')
    #   system_prompt=original_system_prompt
    system_prompt = st.text_area("system prompt",value=system_prompt)
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button(
        "🔥 Ask",
    )

    if ask and prompt:
        results = query_collection(prompt)
        context = results.get("documents")[0]
        relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
        response = call_llm(context=relevant_text, prompt=prompt,system_prompt=system_prompt)
        st.write_stream(response)

        with st.expander("See retrieved documents"):
            st.write(results)

        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)
            st.write(relevant_text)
