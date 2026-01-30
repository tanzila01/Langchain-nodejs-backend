import { Document } from "@langchain/core/documents";
import { crawlLangchainDocsUrls } from "../rag/crawlDocuments.js";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import cliProgress from "cli-progress";

const progressBar = new cliProgress.SingleBar({});

export const loadDocuments = async (): Promise<Document[]> => {
    const langchainDocsUrls = await crawlLangchainDocsUrls();
      console.log(
       `Starting document download. ${langchainDocsUrls.length} total documents.`
     );

    progressBar.start(langchainDocsUrls.length, 0);
    const rawDocuments: Document[] = [];

    for (const url of langchainDocsUrls) {
        const loader = new CheerioWebBaseLoader(url);
        const docs = await loader.load();
        rawDocuments.push(...docs);
        progressBar.increment();
    };

    progressBar.stop();
    console.log(`${rawDocuments.length} documents loaded.`);
    return rawDocuments;
}

const rawDocuments = await loadDocuments();

console.log(rawDocuments.slice(0, 4));