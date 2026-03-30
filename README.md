# ListItemPriceChecker
Checks changes in prices on a list of items extracted from a pdf with item details using LLMs.

Simple project to explore "agentic AI","vibe coding" and "dockerization". Print your amazon lists, input to this program at different days. The program first extracts the item names and prices from these lists using VLMs. Then, it computes if the current price is the lowest observed or not. I did not write a single line of this code as I was mainly curious about what people refer to as "vibe coding". The code contains multiple agents handling different parts of the framework, which are implemented through langgraph. 

A very good improvement to this could have been continuous check every day without the explicit need for printing the list and running the program. However, as Amazon does not allow automated scraping without specific APIs which you need to pay (fairly), I did not go from that route. 
