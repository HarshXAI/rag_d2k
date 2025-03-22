import streamlit as st
import json
import os
import tempfile
import base64
from datetime import datetime
from io import BytesIO
import markdown
from xhtml2pdf import pisa
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from enhanced_financial_data_extractor import extract_financial_data_rag
# At the top of your Streamlit app
import os
import pytesseract

# Set Tesseract path for Streamlit Cloud
if os.path.exists('/usr/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
# Configure Gemini API
GEMINI_API_KEY = "AIzaSyDtQ049lTGHZZoBkBV2wJIJmvTYgQYN0Og"
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY  # Set environment variable for langchain

# Configure page
st.set_page_config(
    page_title="Financial Document Chatbot",
    page_icon="📊",
    layout="wide"
)

# Add speech recognition functionality
def add_voice_input_js():
    st.markdown("""
    <script>
    // Function to add microphone button - improved with better targeting and retry
    function addMicrophoneButton() {
        // Check if button already exists
        if (document.querySelector('#voice-record-btn')) {
            return;
        }
        
        // Find the chat input container
        const chatInputContainer = document.querySelector('.stChatInput');
        if (!chatInputContainer) {
            // Retry after a short delay if not found
            setTimeout(addMicrophoneButton, 500);
            return;
        }
        
        const recordButton = document.createElement("button");
        recordButton.innerHTML = "🎤 Voice";
        recordButton.id = "voice-record-btn";
        recordButton.style = "background-color: #FF4B4B; color: white; border-radius: 20px; padding: 0.5rem 1rem; margin-right: 10px; border: none; position: relative; z-index: 1000; cursor: pointer;";
        
        let isRecording = false;
        let mediaRecorder;
        let audioChunks = [];
        
        recordButton.onclick = function() {
            if (!isRecording) {
                // Start recording
                navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    isRecording = true;
                    recordButton.innerHTML = "🔴 Stop";
                    recordButton.style.backgroundColor = "#FF0000";
                    
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = (event) => {
                        audioChunks.push(event.data);
                    };
                    
                    mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        const reader = new FileReader();
                        reader.readAsDataURL(audioBlob);
                        reader.onloadend = () => {
                            const base64data = reader.result.split(',')[1];
                            
                            // Use session storage to pass data
                            window.sessionStorage.setItem('voiceData', JSON.stringify({
                                audio_data: base64data,
                                type: 'webm'
                            }));
                            window.dispatchEvent(new Event('voiceDataReady'));

                            // Update chat input field
                            const chatInput = document.querySelector('.stChatInput input');
                            if (chatInput) {
                                chatInput.value = "[Voice Message Processing...]";
                                // Trigger React's onChange
                                const event = new Event('input', { bubbles: true });
                                chatInput.dispatchEvent(event);
                                
                                // Trigger a keypress event to submit the message
                                setTimeout(() => {
                                    const enterEvent = new KeyboardEvent('keypress', {
                                        key: 'Enter',
                                        code: 'Enter',
                                        keyCode: 13,
                                        which: 13,
                                        bubbles: true
                                    });
                                    chatInput.dispatchEvent(enterEvent);
                                }, 500);
                            }
                        };
                        audioChunks = [];
                    };
                    
                    mediaRecorder.start();
                })
                .catch(error => {
                    console.error("Error accessing microphone:", error);
                    alert("Error accessing microphone. Please ensure you have granted permission.");
                    isRecording = false;
                });
            } else {
                // Stop recording
                isRecording = false;
                recordButton.innerHTML = "🎤 Voice";
                recordButton.style.backgroundColor = "#FF4B4B";
                if (mediaRecorder && mediaRecorder.state !== "inactive") {
                    mediaRecorder.stop();
                }
            }
            return false; // Prevent default button behavior
        };
        
        // Insert the button next to the chat input field
        const textareaContainer = chatInputContainer.querySelector("div[data-testid='stChatInput'] > div");
        if (textareaContainer) {
            textareaContainer.style.display = "flex";
            textareaContainer.style.alignItems = "center";
            textareaContainer.insertBefore(recordButton, textareaContainer.firstChild);
        } else {
            // Alternative placement if the structure is different
            chatInputContainer.parentNode.insertBefore(recordButton, chatInputContainer);
        }
    }
    
    // Run the function immediately and also on DOM changes
    document.addEventListener('DOMContentLoaded', function() {
        // Initial attempt
        addMicrophoneButton();
        
        // Watch for DOM changes to handle dynamic UI updates
        const observer = new MutationObserver((mutations) => {
            if (!document.querySelector('#voice-record-btn')) {
                addMicrophoneButton();
            }
        });
        
        observer.observe(document.body, { 
            childList: true, 
            subtree: true 
        });
    });
    
    // Also try after a delay in case Streamlit is slow to render
    setTimeout(addMicrophoneButton, 2000);
    </script>
    """, unsafe_allow_html=True)

# Function to process voice data
def handle_voice_input():
    voice_text = ""
    # Check if voice data is available in session state
    if 'voice_data' in st.session_state and st.session_state.voice_data:
        # Process the voice data using Google's Speech-to-Text API
        try:
            from google.cloud import speech
            
            # Initialize the client
            client = speech.SpeechClient.from_service_account_json('google_credentials.json')
            
            # Decode the base64 audio
            audio_bytes = base64.b64decode(st.session_state.voice_data['audio_data'])
            
            # Configure the speech recognition request
            audio = speech.RecognitionAudio(content=audio_bytes)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                sample_rate_hertz=48000,
                language_code="en-US",
                model="default",
                enable_automatic_punctuation=True,
            )
            
            # Send the request
            response = client.recognize(config=config, audio=audio)
            
            # Extract the recognized text
            for result in response.results:
                voice_text += result.alternatives[0].transcript
                
            # Clear the voice data
            st.session_state.voice_data = None
            
        except Exception as e:
            st.error(f"Error processing voice: {str(e)}")
            st.info("Please ensure you have set up Google Speech-to-Text API credentials.")
            voice_text = ""
            
    return voice_text

# JavaScript to listen for voice data
def register_voice_data_listener():
    st.markdown("""
    <script>
    // Listen for the custom event
    window.addEventListener('voiceDataReady', function() {
        const voiceData = window.sessionStorage.getItem('voiceData');
        if (voiceData) {
            // Send to Python via Streamlit session state
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: JSON.parse(voiceData)
            }, '*');
            
            // Clear the session storage
            window.sessionStorage.removeItem('voiceData');
        }
    });
    </script>
    """, unsafe_allow_html=True)

def process_pdf(file_path):
    """Process PDF using extraction code"""
    return extract_financial_data_rag(file_path)

def prepare_context_chunks(financial_data):
    """Convert structured financial data into text chunks for RAG with optimized chunking strategy"""
    chunks = []
    metadata = []
    
    # Process tables - Enhanced with more detailed representation
    if 'financial_data' in financial_data:
        for table in financial_data['financial_data']:
            try:
                # Create a detailed representation of the table with clear structure
                chunk = f"TABLE: {table['table_name']} (Page {table.get('page', 'N/A')})\n\n"
                
                # Add more context about table type
                if "Balance Sheet" in table.get('table_name', ''):
                    chunk += "This is a Balance Sheet table showing assets, liabilities and equity.\n"
                elif "Income Statement" in table.get('table_name', ''):
                    chunk += "This is an Income Statement table showing revenue, expenses and profit.\n"
                elif "Cash Flow" in table.get('table_name', ''):
                    chunk += "This is a Cash Flow Statement showing cash movements.\n"
                
                # Add column headers
                columns = [str(col) for col in table.get('columns', [])]
                if columns:
                    chunk += "COLUMNS: " + " | ".join(columns) + "\n\n"
                
                # Add row data with better formatting
                chunk += "DATA ROWS:\n"
                for row in table.get('rows', []):
                    row_items = []
                    for k, v in row.items():
                        # Format numerical values with currency symbol if appropriate
                        if isinstance(v, (int, float)) and k.lower() not in ['year', 'quarter', 'period']:
                            row_items.append(f"{k}: ${v}")
                        else:
                            row_items.append(f"{k}: {v}")
                    chunk += " | ".join(row_items) + "\n"
                
                chunks.append(chunk)
                metadata.append({
                    "type": "table",
                    "name": table.get('table_name', ''),
                    "page": table.get('page', 0)
                })
            except Exception as e:
                st.error(f"Error processing table: {str(e)}")
    
    # Process text sections with enhanced metadata
    if 'contextual_text' in financial_data:
        for text in financial_data['contextual_text']:
            try:
                # Add tags to the chunk for better context
                tags_str = ", ".join(text.get('tags', []))
                prefix = f"FINANCIAL TEXT (Page {text.get('page', 'N/A')})"
                if tags_str:
                    prefix += f" [TOPICS: {tags_str}]"
                prefix += ": "
                
                chunk = prefix + text.get('content', '')
                
                chunks.append(chunk)
                metadata.append({
                    "type": "text",
                    "tags": text.get('tags', []),
                    "page": text.get('page', 0)
                })
            except Exception as e:
                st.error(f"Error processing text: {str(e)}")
    
    # Process notes - Important for financial context
    if 'notes' in financial_data:
        for note in financial_data.get('notes', []):
            try:
                chunk = f"IMPORTANT NOTE (Page {note.get('page', 'N/A')}): {note.get('content', '')}"
                chunks.append(chunk)
                metadata.append({
                    "type": "note",
                    "page": note.get('page', 0)
                })
            except Exception as e:
                st.error(f"Error processing note: {str(e)}")
    
    return chunks, metadata

def setup_embeddings():
    """Set up optimized embeddings for financial text"""
    # Google's free embedding model handles financial text well
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_document",  # Optimized for document retrieval
        title="Financial Document Analysis"  # Adding title for better context
    )

def setup_rag_chain():
    """Set up a RAG chain for document QA with optimized parameters"""
    # Initialize LLM with parameters for detailed responses
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.2,  # Lower temperature for more factual/accurate responses
        top_p=0.95,       # High top_p for more comprehensive answers
        top_k=40,         # Higher top_k for broader vocabulary selection
        max_output_tokens=4096,  # Maximum output length for detailed analysis
        convert_system_message_to_human=True
    )
    
    # Enhanced prompt template with more financial context
    prompt = ChatPromptTemplate.from_template("""
    You're a financial analyst with expertise in interpreting financial documents, including balance sheets, income statements, cash flow statements, and financial notes.

    The following is information extracted from a financial document. Use this context to provide a detailed answer:
    
    {context}
    
    Question: {question}
    
    Format your answer with:
    1. Direct answer to the question with specific numbers/metrics when available
    2. Supporting analysis with financial calculations or formula explanations when relevant
    3. References to specific tables, page numbers, or sections from the document (cite your sources)
    4. Related financial metrics or ratios when applicable
    5. Brief explanation of what the numbers mean in business context
    
    If specific information isn't available in the context, explain what would typically be needed to answer the question completely.
    """)
    
    # Create simple RAG chain
    chain = (
        {"context": lambda x: x["retrieved_docs"], "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def add_to_report(response_data):
    """Add a response to the report with automatic categorization"""
    # Create a unique identifier based on content
    content_hash = hash(response_data["content"])
    
    # Only add if not already in report
    if content_hash not in [hash(entry["content"]) for entry in st.session_state.report["all_entries"]]:
        # Automatically categorize based on detected tags
        tags = response_data.get("tags", [])
        
        # Try to extract tags from sources if not provided directly
        if not tags and "sources" in response_data:
            for source in response_data["sources"]:
                if isinstance(source, dict) and "type" in source:
                    tags.append(source["type"])
        
        # Fallback to financial domain tags - extract from content
        if not tags:
            financial_terms = {
                "Profitability": ["margin", "profit", "earnings", "ebitda", "ebit", "income", "return", "roi", "roe", "roa"],
                "Revenue": ["revenue", "sales", "turnover"],
                "Balance Sheet": ["assets", "liabilities", "equity", "balance"],
                "Cash Flow": ["cash flow", "liquidity", "solvency", "financing"],
                "Risk Analysis": ["risk", "uncertainty", "debt ratio", "leverage"]
            }
            
            content_lower = response_data["content"].lower()
            for category, terms in financial_terms.items():
                if any(term in content_lower for term in terms):
                    tags.append(category)
                    break
        
        # Final fallback
        category = tags[0] if tags else "General"
        
        # Create section if it doesn't exist
        if category not in st.session_state.report["sections"]:
            st.session_state.report["sections"][category] = []
            
        # Add entry with metadata
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": response_data["query"],
            "content": response_data["content"],
            "sources": response_data.get("sources", []),
            "tags": tags
        }
        
        st.session_state.report["sections"][category].append(entry)
        st.session_state.report["all_entries"].append(entry)
        
        # Replace success message with toast notification
        st.toast(f"✅ Added to {category} section!", icon="📥")
        st.rerun()  # Force UI update
    else:
        st.warning("This content is already in the report")

def generate_report_markdown():
    """Generate a structured Markdown report without sources"""
    md_content = "# Financial Analysis Report\n"
    md_content += "### Generated by AI Financial Assistant\n\n"
    
    if "processed_data" in st.session_state and "metadata" in st.session_state.processed_data:
        metadata = st.session_state.processed_data["metadata"]
        md_content += f"**Document:** {metadata.get('source_file', 'Unknown')}\n\n"
        md_content += f"**Analyzed on:** {metadata.get('extraction_timestamp', 'Unknown')}\n\n"
    
    for section, entries in st.session_state.report["sections"].items():
        if section == "General":
            continue
            
        md_content += f"## {section} Analysis\n\n"
        
        for entry in entries:
            md_content += f"### {entry['timestamp']}\n"
            md_content += f"**Question:** {entry['query']}\n\n"
            md_content += f"{entry['content']}\n\n"
            md_content += "---\n\n"
    
    if "General" in st.session_state.report["sections"]:
        md_content += "## Additional Insights\n\n"
        for entry in st.session_state.report["sections"]["General"]:
            md_content += f"### {entry['timestamp']}\n"
            md_content += f"**Question:** {entry['query']}\n\n"
            md_content += f"{entry['content']}\n\n"
            md_content += "---\n\n"
    
    return md_content

def convert_md_to_pdf(md_content):
    """Convert markdown to PDF with proper styling"""
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content)
    
    # Add CSS styling
    styled_html = f"""
    <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; }}
                h2 {{ color: #34495e; }}
                h3 {{ color: #7f8c8d; }}
                .section {{ margin-bottom: 25px; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
    </html>
    """
    
    # Create PDF
    pdf_buffer = BytesIO()
    pisa.CreatePDF(styled_html, dest=pdf_buffer)
    return pdf_buffer.getvalue()

def main():
    st.title("📊 Financial Document Analyzer")
    st.caption("Upload a financial PDF and chat with your document using AI (now with voice commands)")
    
    # Add voice input functionality - moved earlier in the code
    add_voice_input_js()
    register_voice_data_listener()
    
    # Force a minimum height for the chat container to ensure button visibility
    st.markdown("""
        <style>
        .stChatInput {
            min-height: 60px;
            padding-top: 10px !important;
            padding-bottom: 10px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Handle voice input data from JavaScript
    voice_input = handle_voice_input()
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "temp_file" not in st.session_state:
        st.session_state.temp_file = None
    if "voice_data" not in st.session_state:
        st.session_state.voice_data = None
    # Initialize report structure
    if "report" not in st.session_state:
        st.session_state.report = {
            "sections": {},
            "all_entries": []
        }
    
    # Add Report Status Indicator right after the main title
    if st.session_state.report["all_entries"]:
        entries_count = len(st.session_state.report["all_entries"])
        st.sidebar.success(f"📄 Report contains {entries_count} entries across {len(st.session_state.report['sections'])} sections")
    else:
        st.sidebar.info("ℹ️ No entries in report yet - click 'Add to Report' buttons below responses")
    
    # Voice data receiver via Streamlit component
    voice_data = st.text_input("Voice Data", key="voice_data_input", label_visibility="collapsed")
    if voice_data == "[Voice Message Processing...]":
        # This is a signal that voice data is being processed
        st.session_state.voice_processing = True
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This tool analyzes financial documents and lets you ask questions about the content.
        
        **Capabilities:**
        - Extracts tables, text, and notes from financial PDFs
        - Answers questions about financial metrics
        - Identifies key financial ratios
        - Provides analysis based on document content
        """)
        
        if st.session_state.processed_data:
            st.download_button(
                "Download Extracted Data",
                data=json.dumps(st.session_state.processed_data, indent=2),
                file_name="financial_data.json",
                mime="application/json"
            )
            
        # Add report download buttons - Updated to use Markdown and PDF
        if st.session_state.report["all_entries"]:
            st.markdown("## Report Generation")
            md_report = generate_report_markdown()
            pdf_report = convert_md_to_pdf(md_report)
            
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_report,
                file_name="financial_analysis_report.pdf",
                mime="application/pdf",
                help="Download complete analysis report in PDF format"
            )
            
            # Optional: Keep MD download
            st.download_button(
                label="📝 Download Markdown",
                data=md_report,
                file_name="financial_analysis_report.md",
                mime="text/markdown",
                help="Download report in Markdown format"
            )
            
            if st.button("🧹 Clear Report"):
                st.session_state.report = {"sections": {}, "all_entries": []}
                st.rerun()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a financial document", type=['pdf'])
    
    # Process the file
    if uploaded_file and not st.session_state.processed_data:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        st.session_state.temp_file = temp_file.name
        
        # Process the PDF with status indicator
        with st.status("Processing document...", expanded=True) as status:
            try:
                # Extract data
                st.write("Extracting financial data...")
                financial_data = process_pdf(st.session_state.temp_file)
                st.session_state.processed_data = financial_data
                
                # Create chunks
                st.write("Preparing document for analysis...")
                chunks, metadata_list = prepare_context_chunks(financial_data)
                
                # Create vector store with optimized parameters
                st.write("Building knowledge base...")
                
                # Use optimized embeddings for financial data
                embeddings = setup_embeddings()
                
                # Create documents with metadata
                documents = []
                for i, (text, metadata) in enumerate(zip(chunks, metadata_list)):
                    documents.append((text, {"page": metadata["page"], "type": metadata["type"]}))
                
                # Configure FAISS for better retrieval
                vectorstore = FAISS.from_texts(
                    texts=[doc[0] for doc in documents],
                    embedding=embeddings,
                    metadatas=[doc[1] for doc in documents]
                )
                
                st.session_state.vectorstore = vectorstore
                
                # Set up RAG chain
                st.write("Setting up AI assistant...")
                
                status.update(label="Document processed successfully!", state="complete")
                
                # Document summary
                st.success(f"""
                Document processed! Found:
                - {len([t for t in metadata_list if t['type'] == 'table'])} tables
                - {len([t for t in metadata_list if t['type'] == 'text'])} text sections
                - {len([t for t in metadata_list if t['type'] == 'note'])} notes
                """)
                
            except Exception as e:
                status.update(label=f"Error: {str(e)}", state="error")
                st.error(f"Failed to process document: {str(e)}")
    
    # If we have processed data, show the chat interface
    if st.session_state.processed_data and st.session_state.vectorstore:
        # Display chat messages - Modified to include report buttons
        for i, message in enumerate(st.session_state.chat_history):
            # Check if message is a dictionary with 'role' key
            if isinstance(message, dict) and "role" in message:
                role = message["role"]
                content = message.get("content", "")
                sources = message.get("sources", [])
                
                with st.chat_message(role):
                    st.write(content)
                    
                    # Add report button for assistant messages with enhanced visibility
                    if role == "assistant":
                        if st.button(
                            "📥 Add to Report", 
                            key=f"add_report_{i}",
                            help="Add this response to the final report",
                            use_container_width=True,
                            type="primary"
                        ):
                            # Get the previous user message for context
                            query = st.session_state.chat_history[i-1]["content"] if i > 0 else "Unknown question"
                            add_to_report({
                                "query": query,
                                "content": content,
                                "sources": sources,
                                "tags": message.get("tags", [])
                            })
                        
                        # Show sources if available
                        if sources:
                            with st.expander("Sources"):
                                for source in sources:
                                    # Handle different source formats
                                    if isinstance(source, dict):
                                        page = source.get("page", "N/A")
                                        source_type = source.get("type", "content")
                                        text = source.get("text", "")
                                        
                                        st.caption(f"Page {page} - {source_type}")
                                        if text:
                                            st.text(text[:200] + "..." if len(text) > 200 else text)
            elif isinstance(message, tuple) and len(message) >= 2:
                # Handle tuple format (role, content)
                role, content = message[0], message[1]
                sources = message[2] if len(message) > 2 else []
                
                with st.chat_message(role):
                    st.write(content)
                    
                    # Also add report button for assistant messages in tuple format with enhanced visibility
                    if role == "assistant":
                        if st.button(
                            "📥 Add to Report", 
                            key=f"add_report_tuple_{i}",
                            help="Add this response to the final report",
                            use_container_width=True,
                            type="primary"
                        ):
                            # Get the previous user message for context
                            query = st.session_state.chat_history[i-1][1] if i > 0 and isinstance(st.session_state.chat_history[i-1], tuple) else "Unknown question"
                            add_to_report({
                                "query": query,
                                "content": content,
                                "sources": sources,
                                "tags": []
                            })
            else:
                # Skip invalid message formats
                continue
                
        # Handle voice input if available
        query = None
        if voice_input:
            query = voice_input
            st.info(f"Voice detected: {voice_input}")
        else:
            # Regular text input
            query = st.chat_input("Ask about the financial document or use the microphone button")
        
        if query:
            # Display user message
            user_message = {"role": "user", "content": query}
            st.session_state.chat_history.append(user_message)
            
            with st.chat_message("user"):
                st.write(query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document..."):
                    try:
                        # Enhanced retrieval with better parameters
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_kwargs={
                                "k": 25,        # Retrieve more documents
                                "fetch_k": 50,  # Consider more candidates
                                "score_threshold": 0.5  # Set minimum relevance threshold
                            }
                        )
                        retrieved_docs = retriever.invoke(query)
                        
                        # Format retrieved documents for display
                        sources = []
                        retrieved_text = ""
                        
                        for doc in retrieved_docs:
                            source = {
                                "page": doc.metadata.get("page", "N/A"),
                                "type": doc.metadata.get("type", "unknown"),
                                "text": doc.page_content
                            }
                            sources.append(source)
                            retrieved_text += doc.page_content + "\n\n"
                        
                        # Set up and run chain with optimized parameters
                        chain = setup_rag_chain()
                        response = chain.invoke({
                            "question": query,
                            "retrieved_docs": retrieved_text
                        })
                        
                        # Display response
                        st.write(response)
                        
                        # Add to chat history as dictionary
                        assistant_message = {
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        }
                        st.session_state.chat_history.append(assistant_message)
                        
                    except Exception as e:
                        error_message = f"Error generating response: {str(e)}"
                        st.error(error_message)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_message,
                            "sources": []
                        })
    
        # Add Report Preview Section after the chat interface
        if st.session_state.report["all_entries"]:
            st.markdown("## 📋 Report Preview")
            
            for section, entries in st.session_state.report["sections"].items():
                with st.expander(f"{section} ({len(entries)} entries)"):
                    for entry in entries:
                        st.markdown(f"""
                        **{entry['timestamp']}**  
                        **Q:** {entry['query']}  
                        **A:** {entry['content'][:200]}...
                        """)
                        if entry["sources"]:
                            st.caption(f"Sources: {len(entry['sources'])} references")
    
            # Add Floating Report Controls at the bottom - Updated for PDF download
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 2])
            with col1:
                if st.button("🔄 Update Report Preview"):
                    st.rerun()
            with col2:
                md_report = generate_report_markdown()
                st.download_button(
                    label="📄 Download PDF Report",
                    data=convert_md_to_pdf(md_report),
                    file_name="financial_report.pdf",
                    mime="application/pdf"
                )
            with col3:
                st.download_button(
                    label="📝 Download Markdown",
                    data=md_report,
                    file_name="financial_report.md",
                    mime="text/markdown"
                )
    
    # Clean up temporary file when the app closes
    if st.session_state.temp_file and os.path.exists(st.session_state.temp_file):
        try:
            os.unlink(st.session_state.temp_file)
        except:
            pass

if __name__ == "__main__":
    main()