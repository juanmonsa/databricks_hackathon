# Databricks notebook source
!pip install langchain_community
!pip install pdfkit
!pip install openai
!pip install tiktoken
!pip install faiss-cpu
!pip install unstructured
!pip install yfinance

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir -p /tmp/wkhtmltopdf
# MAGIC cd /tmp/wkhtmltopdf
# MAGIC wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.4/wkhtmltox-0.12.4_linux-generic-amd64.tar.xz
# MAGIC tar -xf wkhtmltox-0.12.4_linux-generic-amd64.tar.xz
# MAGIC

# COMMAND ----------

# --- IMPORTS ---
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentType

import pdfkit
import yfinance as yf
from datetime import datetime

# --- API KEY ---
tk = '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'

# --- 1. LOAD DOCUMENTS ---
def load_documents():
    loader = DirectoryLoader('documents', glob='*.txt')
    return loader.load()

# --- 2. CREATE VECTOR INDEX ---
def create_index(documents):
    embeddings = OpenAIEmbeddings(openai_api_key=tk)
    return FAISS.from_documents(documents, embeddings)

# --- 3. LLM INSTANCE ---
llm = ChatOpenAI(model="gpt-4o", openai_api_key=tk)

# --- 4. AI-GENERATED COVER HTML ---
def generate_cover_with_ai(project_title: str, subtitle: str = "", date: str = None) -> str:
    if not date:
        date = datetime.now().strftime("%B %d, %Y")

    prompt = f"""
You are a creative web designer.

Generate only the HTML for a beautiful, elegant cover page for a financial report. The title is \"{project_title}\" and the subtitle is \"{subtitle}\". 
Use a modern layout, centered content, with the following style constraints:
- Use inline CSS
- Use these brand colors: background #efe4e1, main title color #db3e2c, accent border #c8796e
- Include the generation date: {date}
- Make it suitable as the first page of a PDF report.
Do not include <html>, <head> or <body>, just the content block.
"""
    response = llm.predict(prompt)
    response = response[8:-4]
    return response

# --- 5. GENERATE PDF FROM HTML ---
def generate_pdf_from_html(html_input: str, title="Financial Reasonability Report", subtitle="Automatically generated using AI") -> str:
    try:
        binary = "/tmp/wkhtmltopdf/wkhtmltox/bin/wkhtmltopdf"
        config = pdfkit.configuration(wkhtmltopdf=binary)
        options = {
            'dpi': 400,
            'enable-local-file-access': None,
            'page-size': 'A4',
            'encoding': "UTF-8"
        }

        cover = generate_cover_with_ai(title, subtitle)
        full_html = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head><meta charset="UTF-8"><title>{title}</title></head>
        <body style="font-family:Arial;background:#efe4e1;margin:0;padding:40px;">
        <div style="background:white;padding:30px;border-radius:10px;border: 2px solid #c8796e;">
        {cover}
        <div style="page-break-after: always;"></div>
        {html_input}
        </div>
        </body>
        </html>
        '''

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/Workspace/Users/jorge.r@lambdaanalytics.co/Hackaton/report_{timestamp}.pdf"
        pdfkit.from_string(full_html, output_path, configuration=config, options=options)
        return f"✅ PDF successfully generated: {output_path}"

    except Exception as e:
        return f"❌ PDF generation error: {e}"

# --- 6. RAG TOOL ---
def create_rag_tool(vectorstore):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return Tool(
        name="Financial_RAG",
        func=qa_chain.run,
        description="Use this tool to answer questions based on loaded financial documents."
    )

# --- 7. MATH TOOL ---
def math_tool(input: str) -> str:
    try:
        return str(eval(input))
    except Exception as e:
        return f"Calculation error: {e}"

calculator_tool = Tool(
    name="Calculator",
    func=math_tool,
    description="Use this to perform calculations such as sums, differences, percentages, etc."
)

# --- 8. COMPANY INFO TOOL (HTML) ---
def get_company_info_html(ticker: str) -> str:
    try:
        company = yf.Ticker(ticker)
        info = company.info
        def val(k): return info.get(k, "N/A")

        html = f"""
        <div style="padding:20px;">
            <h2 style="color:#db3e2c;">📊 {val('longName')} ({ticker.upper()})</h2>
            <table style="width:100%;border-collapse:collapse;">
                <tr><td style="padding:8px;"><strong>Industry:</strong></td><td>{val('industry')}</td></tr>
                <tr><td style="padding:8px;"><strong>Sector:</strong></td><td>{val('sector')}</td></tr>
                <tr><td style="padding:8px;"><strong>Country:</strong></td><td>{val('country')}</td></tr>
                <tr><td style="padding:8px;"><strong>Market Cap:</strong></td><td>{val('marketCap')}</td></tr>
                <tr><td style="padding:8px;"><strong>Revenue (TTM):</strong></td><td>{val('totalRevenue')}</td></tr>
                <tr><td style="padding:8px;"><strong>Net Income (TTM):</strong></td><td>{val('netIncomeToCommon')}</td></tr>
                <tr><td style="padding:8px;"><strong>52-Week Range:</strong></td><td>{val('fiftyTwoWeekLow')} - {val('fiftyTwoWeekHigh')}</td></tr>
                <tr><td style="padding:8px;"><strong>Forward P/E:</strong></td><td>{val('forwardPE')}</td></tr>
                <tr><td style="padding:8px;"><strong>Dividend Yield:</strong></td><td>{val('dividendYield')}</td></tr>
                <tr><td style="padding:8px;"><strong>Website:</strong></td><td><a href="{val('website')}" target="_blank">{val('website')}</a></td></tr>
            </table>
            <div style="margin-top:20px;">
                <h3 style="color:#c8796e;">📈 Summary</h3>
                <p style="line-height:1.6;">{val('longBusinessSummary')}</p>
            </div>
        </div>
        """
        return html
    except Exception as e:
        return f"<p style='color:red;'>❌ Error fetching company info for {ticker}: {e}</p>"

# --- 9. COMPANY PDF TOOL ---
def company_info_to_pdf(ticker: str) -> str:
    html = get_company_info_html(ticker)
    return generate_pdf_from_html(html, title=f"{ticker.upper()} Financial Report", subtitle="Generated using yfinance + LLM")

company_info_pdf_tool = Tool(
    name="Company_Info_PDF",
    func=company_info_to_pdf,
    description="Retrieves and exports to PDF the financial profile of a public company using its ticker (e.g. AAPL, TSLA, MSFT)."
)

# --- 10. RAG to PDF TOOL ---
def rag_to_pdf_tool(prompt: str) -> str:
    try:
        retriever = index.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        prompt_html = f"""
Answer the following question with step-by-step reasoning. Then generate a report in HTML format using headings, lists, paragraphs, and tables if necessary. 
Use inline styles and the colors #efe4e1, #db3e2c, and #c8796e to make it visually appealing.

Question: {prompt}
"""
        response_html = qa_chain.run(prompt_html)
        return generate_pdf_from_html(response_html)
    except Exception as e:
        return f"❌ RAG to PDF error: {e}"

rag_pdf_tool = Tool(
    name="RAG_to_PDF",
    func=rag_to_pdf_tool,
    description="Generates a PDF using RAG output in styled HTML."
)

# --- 11. AGENT INITIALIZATION ---
def initialize_custom_agent(tools):
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

# --- 12. EXECUTE INTERACTION ---
def run_agent_interaction(agent, user_message, history=[]):
    try:
        response = agent.run(user_message)
        history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response}
        ])
        return response, history
    except Exception as e:
        return f"Error: {str(e)}", history

# ================================
# ========== EXECUTION ==========
# ================================

documents = load_documents()
index = create_index(documents)

tools = [
    create_rag_tool(index),
    calculator_tool,
    pdf_tool,
    rag_pdf_tool,
    company_info_pdf_tool
]

agent = initialize_custom_agent(tools)

user_message = (
    """Answer: 1) Which department contributed the most to net income growth between 2023 and 2024, and why?
2) If we had to reduce costs by 10% company-wide, which area has the highest opportunity for cost savings without significantly affecting revenue?
3) How did each department's financial performance affect the company's overall profitability margin in 2024?
4) If the Sales department had a 5% increase in revenue next year, how would that affect company-wide net income, assuming costs remain constant?
5) Use Company_Info_PDF to generate a financial report for AAPL and analyze its financial status, if I become his new CFO where do I have to focus to improve their results? analyze their weakness step by step.
    Create a PDF with the analysis using the colors #efe4e1, and #c8796e when need to focus on something for a beautiful result. write at least 200 words per question and always is numbers to talk about your analysis."""
)

history = []
response, history = run_agent_interaction(agent, user_message, history)

print("\n🔍 Agent Response:")
print(response)