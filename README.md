# 🎓 CCMT Chatbot — Your Personal Counselling Assistant
its live check now ! 👉 Try it here: [https://ccmtchatbot.streamlit.app](https://ccmtchatbot.streamlit.app)
Are you confused about CCMT counselling rules, fees, or rounds?  
You're not alone — and that's why I built this free chatbot 💬

**CCMT Chatbot** is an AI-powered assistant that answers your queries using the **official CCMT Information Brochure**, **Fee Tables**, and **Flowcharts**. It’s designed to save you from endless scrolling and confusing documents.

![ccmtsrc](https://github.com/user-attachments/assets/b1df0d2c-9a08-42a4-80e7-a60dc72be9f8)


---

## 💡 Features

- 📖 Answers based only on official CCMT documents
- 💰 Gives exact fee breakdowns
- 🔁 Explains counselling rounds and special scenarios
- 🧠 Built using LangChain, Gemini AI, Pinecone & Streamlit
- 🕒 Made in under 24 hours 💪

---

## 🧪 Tech Stack

| Tool         | Purpose                           |
|--------------|------------------------------------|
| `Streamlit`  | UI for the chatbot                 |
| `LangChain`  | Handles prompt chaining + RAG      |
| `Pinecone`   | Vector database for document search|
| `Gemini API` | Language model for generating answers |
| `Python 3.10`| Runtime language                   |

---

## 🚀 How to Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/MissLostCodes/ccmt-chatbot.git
cd ccmt-chatbot

## 📦 Install Requirements

pip install -r requirements.txt
## Set Up Environment Variables
Create a .env file:

GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_REGION=....from your pinecone index
PINECONE_CLOUD=....from your pinecone index

## ▶️ Run the App
streamlit run app.py


##🛠 Contributing
I'd love your help to make this chatbot better! You can:

Add more data sources

Improve the chatbot’s accuracy

Enhance the UI

Fix bugs or edge cases

Steps to Contribute
Fork this repo

Create a new branch
git checkout -b feature-name

Make your changes

Commit and push
git push origin feature-name

Open a Pull Request ✅

If you’re new to open source, feel free to ask questions. I’d be happy to help!

##🧠 Limitations
This is an early version (MVP) — built in just 1 day

May miss edge cases or very specific queries

All answers are limited to what’s written in official docs

##💬 Feedback
Spotted a bug? Have a suggestion? Open an Issue or send me a message.

##🤍 Built by Shagun Gupta
I made this for students just like you and me .
Please share it with your friends, and feel free to contribute!









