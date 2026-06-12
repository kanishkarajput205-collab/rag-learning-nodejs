import fs from "fs";
import pdfParse from "pdf-parse/lib/pdf-parse.js";

import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

async function loadPdfText(pdfPath) {
  console.log(`Reading PDF: ${pdfPath}`);

  if (!fs.existsSync(pdfPath)) {
    throw new Error(`PDF file not found: ${pdfPath}`);
  }

  const dataBuffer = fs.readFileSync(pdfPath);
  const data = await pdfParse(dataBuffer);

  console.log(`PDF pages: ${data.numpages}`);
  console.log(`Extracted text length: ${data.text.length}`);

  return data.text;
}

async function main() {
  console.log("=== Multi-Modal RAG using Node.js + Ollama ===\n");
  console.log("Note: This simplified version extracts text from PDF only.\n");

  const pdfPath = "docs/attention-is-all-you-need.pdf";

  const pdfText = await loadPdfText(pdfPath);

  const documents = [
    new Document({
      pageContent: pdfText,
      metadata: {
        source: pdfPath,
        type: "pdf-text",
      },
    }),
  ];

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 150,
  });

  const chunks = await splitter.splitDocuments(documents);

  console.log(`Total chunks created: ${chunks.length}`);

  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434",
  });

  console.log("Creating vector store...");
  const vectorStore = await MemoryVectorStore.fromDocuments(
    chunks.slice(0, 80),
    embeddings
  );

  const query =
    "How many attention heads does the Transformer use, and what is the dimension of each head?";

  const relevantDocs = await vectorStore.similaritySearch(query, 3);

  console.log(`\nQuery: ${query}`);
  console.log("\n--- Retrieved Context ---\n");

  relevantDocs.forEach((doc, index) => {
    console.log(`Document ${index + 1}:`);
    console.log(doc.pageContent.slice(0, 1000));
    console.log("\n" + "-".repeat(50) + "\n");
  });

  const context = relevantDocs.map((doc) => doc.pageContent).join("\n");

  const model = new ChatOllama({
    model: "llama3",
    baseUrl: "http://localhost:11434",
    temperature: 0,
  });

  const prompt = `
Answer the question using only the context below.

Question:
${query}

Context:
${context}

If the answer is not available in the context, say:
"I don't have enough information to answer that question based on the provided documents."
`;

  console.log("--- Final Answer ---\n");

  const response = await model.invoke(prompt);
  console.log(response.content);
}

main().catch((error) => {
  console.error("❌ Error:", error.message);
});