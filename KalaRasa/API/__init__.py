def __init__(self, data_dir: str = "data"):
    self.informal_map = self._load_json(
        f"{data_dir}/informal_map.json", 
        fallback=self.INFORMAL_MAP
    )
    self.synonym_map = self._load_json(
        f"{data_dir}/synonyms/", 
        fallback={}
    )