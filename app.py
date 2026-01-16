import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# LangChain & Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. 페이지 및 API 설정
st.set_page_config(page_title="무조건 이기는 가위바위보", page_icon="✌️")

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except FileNotFoundError:
    st.error("Secrets 키를 찾을 수 없습니다. .streamlit/secrets.toml을 확인하세요.")
    st.stop()

st.title("🤖 절대 승리: 가위바위보 봇")
st.markdown("웹캠을 켜고 가위, 바위, 보 중 하나를 내보세요. AI가 당신을 무조건 이깁니다.")

# 2. 리소스 로드 (모델 & RAG 데이터)
@st.cache_resource
def load_teachable_machine_model():
    # Teachable Machine에서 Export한 keras_model.h5 파일 필요
    try:
        model = tf.keras.models.load_model("keras_model.h5", compile=False)
        with open("labels.txt", "r") as f:
            class_names = [line.strip().split(" ")[1] for line in f.readlines()]
        return model, class_names
    except Exception as e:
        st.error(f"모델 로드 실패: {e}. 'keras_model.h5'와 'labels.txt'가 있는지 확인하세요.")
        return None, None

@st.cache_resource
def setup_rag_system():
    # RAG를 위한 가상의 '필승 전략' 문서 생성
    strategy_text = """
    가위바위보 필승 전략 가이드:
    1. 상대가 '가위(Scissors)'를 낼 경우:
       - 전략: '바위(Rock)'를 내야 한다. 바위는 가위를 부순다.
       - 멘트: "당신의 날카로운 가위도 제 바위 앞에서는 무용지물이죠!"
    
    2. 상대가 '바위(Rock)'를 낼 경우:
       - 전략: '보(Paper)'를 내야 한다. 보는 바위를 감싼다.
       - 멘트: "단단한 바위군요. 하지만 제가 보자기(Paper)로 감싸버렸습니다."
    
    3. 상대가 '보(Paper)'를 낼 경우:
       - 전략: '가위(Scissors)'를 내야 한다. 가위는 보를 자른다.
       - 멘트: "넓은 마음의 보자기시군요. 제 가위로 싹둑 잘라드리겠습니다!"
       
    4. 공통 승리 멘트:
       - AI는 항상 0.1초 늦게 내기 때문에 무조건 이길 수밖에 없습니다.
       - 인간의 반응속도로는 AI를 이길 수 없습니다.
    """
    
    # 문서 청킹 및 벡터 저장소 생성
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=x) for x in text_splitter.split_text(strategy_text)]
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# 3. 모델 및 RAG 초기화
model, class_names = load_teachable_machine_model()
vectorstore = setup_rag_system()

# 4. Gemini 2.5 Flash RAG 체인 구성
def get_winning_comment(user_move, ai_move, vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7
    )
    
    retriever = vectorstore.as_retriever()
    
    system_prompt = (
        "당신은 가위바위보 게임의 절대 고수 AI입니다. "
        "아래 제공된 [전략 문맥]을 참고하여, 현재 상황에 맞는 재치있고 약간은 건방진 승리 멘트를 작성하세요. "
        "상대가 {user_move}를 냈고, 당신이 {ai_move}를 내서 이겼습니다."
        "\n\n[전략 문맥]:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "상대: {user_move}, 나: {ai_move}. 승리 멘트를 한 문장으로 해줘.")
    ])
    
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    response = chain.invoke({"user_move": user_move, "ai_move": ai_move, "input": ""})
    return response["answer"]

# 5. 메인 UI 및 게임 로직
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 카메라 입력")
    img_file_buffer = st.camera_input("손을 보여주세요")

if img_file_buffer is not None and model is not None:
    # 이미지 전처리 (Teachable Machine 표준: 224x224, Normalized)
    image = Image.open(img_file_buffer)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # 예측
    prediction = model.predict(data)
    index = np.argmax(prediction)
    user_move_en = class_names[index] # 예: "Rock", "Paper", "Scissors" (라벨링에 따라 다름)
    
    # 한글 매핑 및 승리 로직 (무조건 이기는 로직)
    move_map = {"Rock": "바위", "Paper": "보", "Scissors": "가위"}
    # 라벨 파일이 0 Rock, 1 Paper, 2 Scissors 순서라고 가정 (다를 경우 labels.txt 순서 확인 필요)
    # 안전장치: 라벨 텍스트에 포함된 단어로 매핑
    
    user_move_kr = "알 수 없음"
    ai_move_kr = "알 수 없음"
    
    if "Rock" in user_move_en or "0" in str(index): # 바위
        user_move_kr = "바위"
        ai_move_kr = "보" # 필승
    elif "Paper" in user_move_en or "1" in str(index): # 보
        user_move_kr = "보"
        ai_move_kr = "가위" # 필승
    elif "Scissors" in user_move_en or "2" in str(index): # 가위
        user_move_kr = "가위"
        ai_move_kr = "바위" # 필승

    # 결과 화면 출력
    with col2:
        st.subheader("🎮 게임 결과")
        st.info(f"당신: **{user_move_kr}**")
        st.success(f"AI: **{ai_move_kr}** (승리!)")
        
        # 채팅 인터페이스로 RAG 결과 출력
        st.divider()
        st.write("💬 **AI의 코멘트:**")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # RAG를 통한 멘트 생성
        if user_move_kr != "알 수 없음":
            with st.spinner("AI가 승리 대사를 생각 중입니다..."):
                # 같은 수에 대해 중복 호출 방지를 위해 세션 상태 활용 가능하나, 여기선 매번 생성
                winning_msg = get_winning_comment(user_move_kr, ai_move_kr, vectorstore)
                
                # 채팅 UI에 추가
                with st.chat_message("assistant"):
                    st.write(winning_msg)
