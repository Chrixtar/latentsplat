from typing import Any, Dict, Union


RecursiveStringDict = Dict[str, Union[Any, 'RecursiveStringDict']]

def flatten_nested_string_dict(
    rsd: RecursiveStringDict, 
    prefix: str | None = None
) -> Dict[str, Any]:
    res = {}
    for key, value in rsd.items():
        flattened_key = f"{prefix}/{key}" if prefix is not None else key
        if isinstance(value, dict):
            res.update(flatten_nested_string_dict(value, prefix=flattened_key))
        else:
            res[flattened_key] = value
    return res
