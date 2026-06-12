import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const teslaText = `Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly.`;

async function main() {
  console.log("\n" + "=".repeat(60));
  console.log("RECURSIVE CHARACTER TEXT SPLITTER SOLUTION");
  console.log("=".repeat(60));

  const recursiveSplitter = new RecursiveCharacterTextSplitter({
    separators: ["\n\n", "\n", ". ", " ", ""],
    chunkSize: 100,
    chunkOverlap: 0,
  });

  const chunks = await recursiveSplitter.splitText(teslaText);

  console.log("Same problem text, but with RecursiveCharacterTextSplitter:\n");

  chunks.forEach((chunk, index) => {
    console.log(`Chunk ${index + 1}: (${chunk.length} chars)`);
    console.log(`"${chunk}"`);
    console.log();
  });
}

main().catch((error) => {
  console.error("❌ Error:", error.message);
});