import fs from "fs";
import path from "path";
import readline from "readline";

import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

const chatHistory = [];

async function buildVectorStore() {
  const filePath = path.join("docs", "Microsoft.txt");
  const content = fs.readFileSync(filePath, "utf-8");

  const documents = [
    new Document({
      pageContent: content,
      metadata: { source: filePath },
    }),
  ];

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 700,
    chunkOverlap: 100,
  });

  const chunks = await splitter.splitDocuments(documents);

  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434",
  });

  return await MemoryVectorStore.fromDocuments(chunks.slice(0, 120), embeddings);
}

async function askQuestion(userQuestion, vectorStore, model) {
  console.log(`\n--- You asked: ${userQuestion} ---`);

  let searchQuestion = userQuestion;

  if (chatHistory.length > 0) {
    const historyText = chatHistory
      .map((msg) => `${msg.role}: ${msg.content}`)
      .join("\n");

    const rewritePrompt = `
Given the chat history, rewrite the new question as a standalone searchable question.

Chat history:
${historyText}

New question:
${userQuestion}

Only return the rewritten question.
`;

    const rewritten = await model.invoke(rewritePrompt);
    searchQuestion = rewritten.content.trim();

    console.log(`Searching for: ${searchQuestion}`);
  }

  const docs = await vectorStore.similaritySearch(searchQuestion, 3);

  console.log(`Found ${docs.length} relevant documents:`);
  docs.forEach((doc, index) => {
    const preview = doc.pageContent.split("\n").slice(0, 2).join("\n");
    console.log(`Doc ${index + 1}: ${preview}...`);
  });

  const context = docs.map((doc) => `- ${doc.pageContent}`).join("\n");

  const finalPrompt = `
Answer the user's question using only the provided documents and conversation history.

Question:
${userQuestion}

Documents:
${context}

If the answer is not available in the documents, say:
"I don't have enough information to answer that question based on the provided documents."
`;

  const answerResult = await model.invoke(finalPrompt);
  const answer = answerResult.content;

  chatHistory.push({ role: "Human", content: userQuestion });
  chatHistory.push({ role: "AI", content: answer });

  console.log(`\nAnswer: ${answer}`);
}

async function main() {
  console.log("=== History Aware RAG using Node.js + Ollama ===");
  console.log("Ask me questions. Type 'quit' to exit.\n");

  const vectorStore = await buildVectorStore();

  const model = new ChatOllama({
    model: "llama3",
    baseUrl: "http://localhost:11434",
    temperature: 0,
  });

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  function askInput() {
    rl.question("\nYour question: ", async (question) => {
      if (question.toLowerCase() === "quit") {
        console.log("Goodbye!");
        rl.close();
        return;
      }

      await askQuestion(question, vectorStore, model);
      askInput();
    });
  }

  askInput();
}

main().catch((error) => {
  console.error("❌ Error:", error.message);
});
