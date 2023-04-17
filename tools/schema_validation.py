import jsonschema as jss, jsonschema.exceptions as jse
from copy import deepcopy

kin_pattern_properties = {
    "^.+$": {
        "type": "object",
        "required": ["Uniprot Accession ID", "Gene Name"],
        "properties": {
            "Uniprot Accession ID": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "Gene Name": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        },
        "additionalProperties": True,
    }
}

site_pattern_properties = deepcopy(kin_pattern_properties)
site_pattern_properties["^.+$"]["properties"].update(
    {"Location": {"type": "array", "minItems": 1, "items": {"type": "string"}}}
)
site_pattern_properties["^.+$"]["required"].append("Location")


KinSchema = {
    "type": "object",
    "minProperties": 1,
    "patternProperties": kin_pattern_properties,
    "additionalProperties": False,
}

SiteSchema = deepcopy(KinSchema)
SiteSchema["patternProperties"] = site_pattern_properties

bypass_gc = {
    "Known Group": {
        "type": "string",
        # "enum": ["<UNANNOTATED>", "ATYPICAL", "AGC", "CAMK", "CK1", "CMGC", "OTHER", "STE", "TK", "TKL"],
    }
}

SiteSchemaBypassGC = deepcopy(SiteSchema)
SiteSchemaBypassGC["patternProperties"]["^.+$"]["properties"].update(bypass_gc)

if __name__ == "__main__":
    sample = {
        "ABCDEFGHIJKLMNO": {"Uniprot Accession ID": ["P12345"], "Gene Name": ["ABC1"]},
        "PQRSTUVWXYZAABB": {"Uniprot Accession ID": ["P98765", "Q55555"], "Gene Name": ["PQR2", "XYZ3"]},
    }

    bad_sample = {"": {"Uniprot Accession ID": ["P12345"], "Gene Name": ["ABC1"]}}

    bad_sample_2 = {"ABCDEFGHIJKLMNO": {"Uniprot Accession ID": "P12345"}}

    for s in [sample, bad_sample, bad_sample_2]:
        try:
            jss.validate(s, KinSchema)
            print("Valid")
        except jse.ValidationError as e:
            print("Invalid")

    sample = {
        "ABCDEFGHIJKLMNO": {"Uniprot Accession ID": ["P12345"], "Gene Name": ["ABC1"], "Location": ["T777"]},
        "PQRSTUVWXYZAABB": {
            "Uniprot Accession ID": ["P98765", "Q55555"],
            "Gene Name": ["PQR2", "XYZ3"],
            "Location": ["S555", "T101010"],
        },
    }

    bad_sample = {"XXXYXXXXXYXXX": {"Uniprot Accession ID": ["P12345"], "Gene Name": ["ABC1"], "Location": [999]}}

    bad_sample_2 = {
        "ABCDEFGHIJKLMNO": {
            "Uniprot Accession ID": ["P12345"],
            "Gene Name": ["ABC1"],
        }
    }

    for s in [sample, bad_sample, bad_sample_2]:
        try:
            jss.validate(s, SiteSchema)
            print("Valid")
        except jse.ValidationError as e:
            print("Invalid")
