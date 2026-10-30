import requests
import streamlit as st


#*********************
# FastAPI Backend    *
#*********************

API_URL = "http://127.0.0.1:8000"


#*********************************
# Streamlit Page Configuration   *
#*********************************

st.set_page_config(
    page_title="AI/ML Research Navigator",
    page_icon="🔬",
    layout="wide",
)


#******************
# Session State   *
#******************

if "question" not in st.session_state:
    st.session_state.question = ""

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""


#******************************
# Example Question Callback   *
#******************************

def select_example_question(example):
    """
    Store the selected example question.

    The callback runs before Streamlit redraws the page.
    """

    st.session_state.selected_question = example


#***********************
# Application Header   *
#***********************

st.title("🔬 AI/ML Research Navigator")

st.markdown(
    """
    Ask questions about the computer vision research papers
    stored in the knowledge base.
    """
)


#************
# Sidebar   *
#************

with st.sidebar:

    st.header("Knowledge Base")

    st.write("Indexed research papers:")

    st.markdown(
        """
        - 📄 **ResNet**
        - 📄 **Vision Transformer (ViT)**
        - 📄 **Swin Transformer**
        """
    )

    st.divider()

    st.header("RAG Pipeline")

    st.markdown(
        """
        - Semantic retrieval
        - Hugging Face embeddings
        - Chroma vector database
        - Qwen LLM via Ollama
        - FastAPI backend
        """
    )

    st.divider()

    st.caption(
        "Answers are generated using only the "
        "retrieved research-paper context."
    )


#*********************
# Example Questions  *
#*********************

st.subheader("Ask a Research Question")

st.caption("Example questions:")

example_questions = [
    "What is the main idea behind residual learning?",
    "How does Vision Transformer process an image?",
    "What is the purpose of shifted windows in Swin Transformer?",
    "What is the role of patch embeddings in ViT?",
]

example_columns = st.columns(4)

for index, example in enumerate(example_questions):

    with example_columns[index]:

        st.button(
            example,
            key=f"example_question_{index}",
            use_container_width=True,
            on_click=select_example_question,
            args=(example,),
        )


#*****************************************
# Update Question From Selected Example  *
#*****************************************

if st.session_state.selected_question:

    st.session_state.question = (
        st.session_state.selected_question
    )

    st.session_state.selected_question = ""


#*******************
# Question Input   *
#*******************

st.text_input(
    "Question",
    key="question",
    placeholder=(
        "e.g. How is Vision Transformer different from ResNet?"
    ),
    label_visibility="collapsed",
)


#*****************
# Ask Question   *
#*****************

if st.button(
    "Ask Question",
    type="primary",
    use_container_width=True,
):

    question = st.session_state.question.strip()


    #**********************
    # Validate Question   *
    #**********************

    if not question:

        st.warning(
            "Please enter a research question."
        )

    else:

        try:

            #***************************
            # Send Request to FastAPI  *
            #***************************

            with st.spinner(
                "Searching research papers and generating an answer..."
            ):

                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question
                    },
                    timeout=120,
                )


            #************************
            # Successful Response   *
            #************************

            if response.status_code == 200:

                data = response.json()

                answer = data.get(
                    "answer",
                    "No answer returned."
                )

                sources = data.get(
                    "sources",
                    []
                )


                #***********
                # Answer   *
                #***********

                st.divider()

                st.subheader("Answer")

                st.markdown(answer)


                #***********
                # Sources  *
                #***********

                st.divider()

                st.subheader("Sources")


                if not sources:

                    st.info(
                        "No sources found."
                    )

                else:

                    source_pages = {}


                    #***********************
                    # Collect unique pages *
                    #***********************

                    for source in sources:

                        filename = source.get(
                            "source",
                            "Unknown"
                        )

                        page = source.get(
                            "page",
                            "Unknown"
                        )

                        if filename not in source_pages:

                            source_pages[filename] = set()

                        source_pages[filename].add(page)


                    #*********************
                    # Display documents  *
                    #*********************

                    for filename in sorted(
                        source_pages.keys()
                    ):

                        pages = source_pages[filename]


                        #**************
                        # Sort pages  *
                        #**************

                        try:

                            sorted_pages = sorted(
                                pages,
                                key=lambda page: int(page)
                            )

                        except (
                            ValueError,
                            TypeError
                        ):

                            sorted_pages = sorted(
                                pages,
                                key=str
                            )


                        pages_text = ", ".join(
                            str(page)
                            for page in sorted_pages
                        )


                        #******************
                        # Display source  *
                        #******************

                        with st.expander(
                            f"📄 {filename}"
                        ):

                            st.write(
                                f"Pages: {pages_text}"
                            )


            #*************
            # API Error  *
            #*************
            else:

                st.error(
                    f"API request failed "
                    f"(HTTP {response.status_code})."
                )

                with st.expander(
                    "View API response"
                ):

                    st.code(
                        response.text
                    )


        #****************************
        # FastAPI Connection Error  *
        #****************************

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the FastAPI server."
            )

            st.info(
                "Make sure FastAPI is running on "
                "http://127.0.0.1:8000"
            )


        #*******************
        # Request Timeout  *
        #*******************

        except requests.exceptions.Timeout:

            st.error(
                "The request timed out."
            )

            st.info(
                "The local LLM may still be processing "
                "the question. Please try again."
            )


        #********************
        # Unexpected Error  *
        #********************

        except Exception as e:

            st.error(
                f"Unexpected error: {e}"
            )