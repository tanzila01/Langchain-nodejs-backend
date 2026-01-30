import dotenv from "dotenv";
import { loadDocuments } from "./loadDocuments";
import { splitDocuments } from "./splitDocuments";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import cliProgress from "cli-progress";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

dotenv.config();

const rawDocuments = await loadDocuments();

const chunkedDocuments = await splitDocuments(rawDocuments);

// Google Gemini Embeddings LLM
const embeddingLLM = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001",
});

const pinecone = new Pinecone();
const pineconeIndex = pinecone.index("genai-js");

console.log("Starting Vectorization...");

// More robust filtering
const filteredDocuments = chunkedDocuments.filter((doc) => {
      console.log("page content",doc.pageContent)
  // Check for valid pageContent
  if (!doc.pageContent || typeof doc.pageContent !== "string") {
    return false;
  }
  
  // Check for minimum length (adjust as needed)
  const trimmedContent = doc.pageContent.trim();
  console.log(`Trimmed Content: "${trimmedContent}"`);
  if (trimmedContent.length === 0) {
    return false;
  }
  
  // Optional: Check for minimum word count
  const wordCount = trimmedContent.split(/\s+/).length;
  console.log(`Word Count: ${wordCount}`);
  if (wordCount < 3) {
    return false;
  }
  
  return true;
});

console.log(
  `Filtered out ${chunkedDocuments.length - filteredDocuments.length} invalid documents`
);
console.log(`Processing ${filteredDocuments.length} valid documents`);

if (filteredDocuments.length === 0) {
  console.error("No valid documents to process!");
  process.exit(1);
}

const progressBar = new cliProgress.SingleBar({});
progressBar.start(filteredDocuments.length, 0);

// Process in batches to respect rate limits and catch errors
const BATCH_SIZE = 100;
const BATCH_DELAY = 90000; // 1.5 minutes for Google Gemini free tier

for (let i = 0; i < filteredDocuments.length; i += BATCH_SIZE) {
  const batch = filteredDocuments.slice(i, i + BATCH_SIZE);
  
  try {
    // Wait before processing (except first batch)
    if (i > 0) {
      console.log(`\nWaiting 90 seconds before next batch...`);
      await new Promise(resolve => setTimeout(resolve, BATCH_DELAY));
    }
    
    console.log(`\nProcessing batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(filteredDocuments.length / BATCH_SIZE)}`);
    
    await PineconeStore.fromDocuments(batch, embeddingLLM, {
      pineconeIndex,
    });
    
    progressBar.increment(batch.length);
    
  } catch (error) {
    console.error(`\nError processing batch starting at index ${i}:`, error);
    
    // Log problematic documents for debugging
    console.log("\nProblematic documents in this batch:");
    batch.forEach((doc, idx) => {
      console.log(`Document ${i + idx}:`, {
        contentLength: doc.pageContent?.length || 0,
        preview: doc.pageContent?.substring(0, 100) || "EMPTY",
        metadata: doc.metadata
      });
    });
    
    // Decide whether to continue or stop
    throw error; // Or use 'continue' to skip failed batches
  }
}

progressBar.stop();
console.log("\n✓ Chunked documents stored in Pinecone.");