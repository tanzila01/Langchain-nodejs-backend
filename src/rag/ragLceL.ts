import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createRetriever } from "./retriever";
import { RunnableSequence } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "@langchain/classic/util/document";
import { ChatHandler, chat } from "../utils/chat";
import dotenv from "dotenv";
import { Document } from "@langchain/core/documents";
import {ChatGoogleGenerativeAI} from "@langchain/google-genai";

dotenv.config();

const prompt = ChatPromptTemplate.fromMessages([
  [
    "human",
    `You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:`,
  ],
]);

// OpenAI chat LLM
// const llm = new ChatOpenAI({
//   model: "gpt-3.5-turbo",
//   maxTokens: 500,
// });

// Google chat LLM
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",   // if this model is not available anymore then please check another available model from google gemini api documentation
  
  // Important Note: There is a problem with maxOutputTokens parameters in google langchain package, it causes error
  // Don't use it for now, I will update the code in future if langchain team fixes this issue
  // maxOutputTokens: 500,
});

const outputParser = new StringOutputParser();

const retriever = await createRetriever();

const retrievalChain = RunnableSequence.from([
  (input) => input.question,
  retriever,
  formatDocumentsAsString,

  // Langchain has moved 'formatDocumentsAsString' to classic package in version 1.0, which means it may not be maintained in long term
  // for real life project, use the following code to convert chunk Documents to string
  // (docs: Document[]) => docs.map((doc) => doc.pageContent).join("\n\n"),
]);

const generationChain = RunnableSequence.from([
  {
    question: (input) => input.question,
    context: retrievalChain,
  },
  prompt,
  llm,
  outputParser,
]);

console.log("before run function...");

// const run = async (question: string) => {
//   const result = await generationChain.stream({
//     question: question,
//   });
//   console.log("Result:", result);
// };

const run = async (question: string) => {
  const stream = await generationChain.stream({
    question: question,
  });

  let finalAnswer = "";

  for await (const chunk of stream) {
    // Each chunk is usually a string because of StringOutputParser
    if (typeof chunk === "string") {
      process.stdout.write(chunk);   // stream to terminal
      finalAnswer += chunk;
    } else {
      // In some cases, chunk can be an object
      console.log("Chunk:", chunk);
    }
  }

  console.log("\n\nFinal Answer:\n", finalAnswer);
};
run("What is lang chain?");

// const chatHandler: ChatHandler = async (question: string) => {
//   return {
//     answer: generationChain.stream({
//       question,
//     }),
//   };
// };

// chat(chatHandler);
