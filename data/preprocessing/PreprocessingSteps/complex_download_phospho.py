import re, requests, io, pandas as pd, tqdm, aiohttp, asyncio, json

async def complex_download_phospho():
	global pb, lock
	url = "http://kinase.com/web/current/kinbase/"
	r = requests.get(url)
	if r.status_code != 200:
		raise requests.HTTPError(f"Error downloading organisms from kinase.com web page:\n{r.text}")
	
	pattern = open("./complex_organism_phospho_regex.txt", "r").read()
	available_organisms_raw = re.findall(pattern, r.text)[0]
	available_organisms_raw = [re.sub(r"\s+", "", x) for x in available_organisms_raw.split("<li>")][1:]
	available_organisms_to_url = {re.findall(r"<em>(.+)<\/em>", re.sub(r"\s+", "", x))[0]: f"http://kinase.com/web/current/kinbase/genes/speciesID/{re.findall('/SpeciesID/([0-9]+)/', x)[0]}/" for x in available_organisms_raw}
	print(f"Found {len(available_organisms_to_url)} organisms: {', '.join(available_organisms_to_url.keys())}")
	pb = tqdm.tqdm(total = len(available_organisms_to_url), colour = 'cyan')
	lock = asyncio.Lock()
	async with aiohttp.ClientSession() as session:
		ret = await asyncio.gather(*[async_get(url, session) for url in available_organisms_to_url.values()])
		ret = {k: v for k, v in zip(available_organisms_to_url.keys(), ret)}
	
	tabs = [pd.read_html(page)[0] for page in ret.values()]
	result_df = pd.concat(tabs, axis = 0, ignore_index=True).drop('Select', axis = 1)

async def async_get(url, session):
	global pb, lock
	try:
		async with session.get(url=url) as r:
			if r.status != 200:
				raise requests.HTTPError(f"Error downloading kinase.com web page:\n{await r.text()}")
			async with lock:
				pb.update(1)
			return await r.text()

	except Exception as e:
		print("Unable to get url {} due to {} --- {}.".format(url, e.__class__.__name__, e))


if __name__ == "__main__":
	asyncio.run(complex_download_phospho())