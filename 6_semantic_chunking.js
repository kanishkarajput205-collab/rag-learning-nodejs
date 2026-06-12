import { OllamaEmbeddings } from "@langchain/ollama";

const teslaText = `Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs.`;

function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function main() {
  console.log("SEMANTIC CHUNKING RESULTS");
  console.log("=".repeat(50));

  const paragraphs = teslaText
    .split("\n\n")
    .map((p) => p.trim())
    .filter(Boolean);

  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434",
  });

  const vectors = await embeddings.embedDocuments(paragraphs);

  const chunks = [];
  let currentChunk = paragraphs[0];

  for (let i = 1; i < paragraphs.length; i++) {
    const similarity = cosineSimilarity(vectors[i - 1], vectors[i]);

    if (similarity > 0.55) {
      currentChunk += "\n\n" + paragraphs[i];
    } else {
      chunks.push(currentChunk);
      currentChunk = paragraphs[i];
    }
  }

  chunks.push(currentChunk);

  chunks.forEach((chunk, index) => {
    console.log(`Chunk ${index + 1}: (${chunk.length} chars)`);
    console.log(`"${chunk}"`);
    console.log();
  });
}

main().catch((error) => {
  console.error("❌ Error:", error.message);
});