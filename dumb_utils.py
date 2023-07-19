import re
import copy
import json
import random
import string
import http.client

import chromadb
import torch
import torch.nn.functional as F

from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel

from pingpong import PingPong
from pingpong.pingpong import PPManager
from pingpong.context.strategy import CtxStrategy

default_instruction = """Below texts come from the webpages that you provided in '{ping}'. Try to explain '{ping}' in detail as much as possible. Your exaplanation should almost based on the text below. Try not to write anything unrelated information.
=====================
"""
    
class URLSearchStrategy(CtxStrategy):
    def __init__(
        self,
        similarity_searcher,
        instruction=default_instruction,
        db_name=None, chunk_size=1000
    ):
        self.searcher = similarity_searcher
        self.instruction = instruction
        self.db_name = db_name
        self.chunk_size = chunk_size

        if self.searcher is None:
            raise ValueError("SimilaritySearcher is not set.")

        if self.db_name is None:
            self.db_name = URLSearchStrategy.id_generator()
        
    def __call__(self, ppmanager: PPManager, urls, top_k=8, max_tokens=1024, keep_original=False):
        ppm = copy.deepcopy(ppmanager)
        last_ping = ppm.pingpongs[-1].ping
        # 1st yield
        ppm.add_pong("![loading](https://i.ibb.co/RPSPL5F/loading.gif)\n")
        ppm.append_pong("• Creating Chroma DB Collection...")
        yield True, ppm, "• Creating Chroma DB Collection √"
        
        chroma_client = chromadb.Client()
        try:
            chroma_client.delete_collection(self.db_name)
        except:
            pass
        
        col = chroma_client.create_collection(self.db_name)
        
        # 2nd yield
        ppm.replace_last_pong("![loading](https://i.ibb.co/RPSPL5F/loading.gif)\n")
        ppm.append_pong("• Creating Chroma DB Collection √\n")
        ppm.append_pong("• URL Searching...\n")
        yield True, ppm, "• URL Searching √"

        # HTML parsing
        search_results = []
        success_urls = []
        for url in urls:
            parse_result, contents = self._parse_html(url)
            if parse_result == True:
                success_urls.append(url)
                search_results.append(contents)
                
                ppm.append_pong(f"    - {url} √\n")
                yield True, ppm, f" ▷ {url} √"

        if len(search_results) == 0:
            yield False, ppm, "There is no valid URLs. Check if there are trailing characters such as .(dot), ,(comma), etc., LLM will answer to your question based on its base knowledge."
                
        if len(' '.join(search_results).split(' ')) < max_tokens:
            final_result = ' '.join(search_results)

            # 3rd yield
            ppm.replace_last_pong("![loading](https://i.ibb.co/RPSPL5F/loading.gif)\n")
            ppm.append_pong("• Creating Chroma DB Collection √\n")
            ppm.append_pong("• URL Searching √\n")
            for url in success_urls:
                ppm.append_pong(f"    - {url} √\n")
            yield True, ppm, "• Done √"

            last_ping = self.instruction.format(ping=last_ping)
            last_ping = last_ping + final_result
            
            ppm.pingpongs[-1].ping = last_ping
            ppm.replace_last_pong("")
            yield True, ppm, "⏳ Wait until LLM generates message for you ⏳"
            
        else:
            # 3rd yield
            ppm.replace_last_pong("![loading](https://i.ibb.co/RPSPL5F/loading.gif)\n")
            ppm.append_pong("• Creating Chroma DB Collection √\n")
            ppm.append_pong("• URL Searching √\n")
            for url in success_urls:
                ppm.append_pong(f"    - {url} √\n")
            ppm.append_pong("• Creating embeddings...")
            yield True, ppm, "• Creating embeddings √"        

            final_chunks = []            
            for search_result in search_results:
                chunks = self._create_chunks(
                    search_result, 
                    chunk_size=self.searcher.max_length
                )
                final_chunks.append(chunks)  

            self._put_chunks_into_collection(
                col, final_chunks, docs_per_step=1
            )

            query_results = self._query(
                col, f"query: {last_ping}", top_k,
            )

            # 4th yield
            ppm.replace_last_pong("![loading](https://i.ibb.co/RPSPL5F/loading.gif)\n")
            ppm.append_pong("• Creating Chroma DB Collection √\n")
            ppm.append_pong("• URL Searching √\n")
            for url in success_urls:
                ppm.append_pong(f"    - {url} √\n")        
            ppm.append_pong("• Creating embeddings √\n")
            ppm.append_pong("• Information retrieval...")
            yield True, ppm, "• Information retrieval √"

            last_ping = self.instruction.format(ping=last_ping)
            for doc in query_results['documents'][0]:
                last_ping = last_ping + doc.replace('passage: ', '') + "\n"

            # 5th yield
            ppm.replace_last_pong("![loading](https://i.ibb.co/RPSPL5F/loading.gif)\n")
            ppm.append_pong("• Creating Chroma DB Collection √\n")
            ppm.append_pong("• URL Searching √\n")
            for url in success_urls:
                ppm.append_pong(f"    - {url} √\n")        
            ppm.append_pong("• Creating embeddings √\n")
            ppm.append_pong("• Information retrieval √")
            yield True, ppm, "• Done √"

            ppm.pingpongs[-1].ping = last_ping
            ppm.replace_last_pong("")
            yield True, ppm, "⏳ Wait until LLM generates message for you ⏳"

    def _parse_html(self, url):
        try: 
            page = urlopen(url, timeout=5)
            html_bytes = page.read()
            html = html_bytes.decode("utf-8")
        except:
            return False, None
        
        text = ""
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup.findAll('p'):
            for string in tag.strings:
                text = text + string
                
        for tag in soup.findAll('pre'):
            for string in tag.strings:
                text = text + string

        text = self._replace_multiple_newlines(text)
        return True, text
    
    def _query(
        self, collection, q, top_k
    ):
        _, q_embeddings_list = self.searcher.get_embeddings([q])

        return collection.query(
            query_embeddings=q_embeddings_list,
            n_results=top_k
        )
    
    # chunk_size == number of characters
    def _create_chunks(self, text, chunk_size):
        chunks = []

        for idx in range(0, len(text), chunk_size):
            chunks.append(
                f"passage: {text[idx:idx+chunk_size]}"
            )

        return chunks
    
    def _put_chunk_into_collection(
        self, collection, chunk_id, chunk, docs_per_step=1
    ):
        for i in range(0, len(chunk), docs_per_step):
            cur_texts = chunk[i:i+docs_per_step]
            _, embeddings_list = self.searcher.get_embeddings(cur_texts)
            ids = [
                f"id-{chunk_id}-{num}" for num in range(i, i+docs_per_step)
            ]

            collection.add(
              embeddings=embeddings_list,
              documents=cur_texts,
              ids=ids
            )

    def _put_chunks_into_collection(
        self, collection,
        chunks, docs_per_step=1
    ):
        for idx, chunk in enumerate(chunks):
            self._put_chunk_into_collection(
                collection, idx, 
                chunk, docs_per_step=docs_per_step
            )

    def _replace_multiple_newlines(self, text):
        """Replaces multiple newline characters with a single newline character."""
        pattern = re.compile(r"\n+")
        return pattern.sub("\n", text)             
            
    @classmethod
    def id_generator(cls, size=10, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))