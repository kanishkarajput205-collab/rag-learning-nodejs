import fs from "fs";
import path from "path";

import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

async function main() {
  console.log("=== Answer Generation using Node.js + Ollama ===\n");

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

  const vectorStore = await MemoryVectorStore.fromDocuments(
    chunks.slice(0, 120),
    embeddings
  );

  const query = "How much did Microsoft pay to acquire GitHub?";

  const relevantDocs = await vectorStore.similaritySearch(query, 5);

  console.log(`User Query: ${query}`);
  console.log("\n--- Context ---\n");

  relevantDocs.forEach((doc, index) => {
    console.log(`Document ${index + 1}:`);
    console.log(doc.pageContent);
    console.log("\n" + "-".repeat(50) + "\n");
  });

  const context = relevantDocs.map((doc) => `- ${doc.pageContent}`).join("\n");

  const prompt = `
Based on the following documents, answer this question:

Question: ${query}

Documents:
${context}

Give a clear answer using only the information from the documents.
If the answer is not present, say:
"I don't have enough information to answer that question based on the provided documents."
`;

  const model = new ChatOllama({
    model: "llama3",
    baseUrl: "http://localhost:11434",
    temperature: 0,
  });

  const result = await model.invoke(prompt);

  console.log("\n--- Generated Response ---");
  console.log(result.content);
}

main().catch((error) => {
  console.error("❌ Error:", error.message);
});