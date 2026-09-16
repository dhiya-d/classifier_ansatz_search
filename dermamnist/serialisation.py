# ----------------------------------------------------------------
# serialisation.py  —  Convert motifs to/from JSON-serialisable dicts
#
# Already length-agnostic: serialises whatever list of layers it is
# given, so variable-length genotypes need no changes here.
# ----------------------------------------------------------------

from hierarqcal import Qcycle, Qmask, Qunmask, Qmotifs


def motif_to_dict(motif) -> dict:
    """
    Serialise a Qmotifs object of any length to a plain dict (JSON-safe).
    Gate functions are stored by name only.
    """
    layer_list = []
    for sub in motif:
        if isinstance(sub, Qcycle):
            layer_list.append({
                "type":          "Qcycle",
                "mapping":       sub.mapping.name if sub.mapping else None,
                "stride":        sub.stride,
                "step":          sub.step,
                "offset":        sub.offset,
                "boundary":      sub.boundary,
                "share_weights": sub.share_weights,
            })
        elif isinstance(sub, Qmask):
            layer_list.append({
                "type":           "Qmask",
                "mapping":        sub.mapping.name if sub.mapping else None,
                "global_pattern": sub.global_pattern,
                "strides":        getattr(sub, "strides",    None),
                "steps":          getattr(sub, "steps",      None),
                "offsets":        getattr(sub, "offsets",    None),
                "boundaries":     getattr(sub, "boundaries", None),
            })
        elif isinstance(sub, Qunmask):
            layer_list.append({
                "type": "Qunmask",
                "arg":  sub.args[0] if hasattr(sub, "args") else "previous",
            })
        else:
            raise ValueError(f"Unknown motif type: {type(sub)}")

    return {"layers": layer_list}


def motif_from_dict(d: dict, mapping_dict: dict):
    """
    Reconstruct a Qmotifs object from a serialised dict.

    Parameters
    ----------
    d            : dict produced by motif_to_dict
    mapping_dict : {name: Qunitary} registry (from gates.MAPPING_DICT)
    """
    layer_list = []
    for layer in d["layers"]:
        t       = layer["type"]
        mapping = (mapping_dict.get(layer["mapping"])
                   if layer.get("mapping") else None)

        if t == "Qcycle":
            layer_list.append(Qcycle(
                mapping      = mapping,
                stride       = layer["stride"],
                step         = layer["step"],
                offset       = layer["offset"],
                boundary     = layer["boundary"],
                share_weights= layer["share_weights"],
            ))
        elif t == "Qmask":
            kwargs = {"global_pattern": layer["global_pattern"]}
            if mapping:                  kwargs["mapping"]    = mapping
            if layer.get("strides"):     kwargs["strides"]    = layer["strides"]
            if layer.get("steps"):       kwargs["steps"]      = layer["steps"]
            if layer.get("offsets"):     kwargs["offsets"]    = layer["offsets"]
            if layer.get("boundaries"):  kwargs["boundaries"] = layer["boundaries"]
            layer_list.append(Qmask(**kwargs))
        elif t == "Qunmask":
            layer_list.append(Qunmask(layer.get("arg", "previous")))
        else:
            raise ValueError(f"Unknown layer type: {t}")

    return Qmotifs(tuple(layer_list))