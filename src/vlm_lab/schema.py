"""Deterministic JSON Schema construction for CORD v2 model output.

This module implements the construction algorithm frozen in
``docs/proposals/phase1_closure_prereg.md`` §6.5 (P-14b). The point of a
generated schema is that it is produced by a fixed rule from a fixed
corpus, rather than hand-authored after looking at the data: the same
corpus always yields the same schema document, and the document can be
hashed and recorded alongside every result artifact.

The schema is used for the *strict schema validity* rate only. Per §6.3 it
never gates content scoring — a parseable but schema-invalid prediction
still receives TED-Acc and field-exact credit.

The corpus is the union of the **train + validation** converted ground
truths (never ``test`` — ADR-008). Producing the actual
``configs/cord_v2_output.schema.json`` document therefore requires loading
the dataset and is deliberately not done here; this module is the
algorithm and nothing else.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

import jsonschema

JSON_SCHEMA_DRAFT_URI = "https://json-schema.org/draft/2020-12/schema"

_ROOT_PATH = "$"
_SUPPORTED_SHAPES = ("string", "object", "array")

# Exact-type lookup rather than isinstance, so bool is reported as "boolean"
# instead of matching int.
_SHAPE_NAMES_BY_TYPE = {
    str: "string",
    dict: "object",
    list: "array",
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
}


class SchemaShapeError(ValueError):
    """A corpus path holds values whose shape the §6.5 algorithm cannot encode.

    Raised for heterogeneous shapes at one path (objects mixed with arrays,
    strings mixed with objects, ...) and for unsupported shapes (a non-string
    scalar, ``null``, an array of scalars). This is deliberately a hard
    failure: silently degrading such a path to a string leaf produces a
    schema that rejects the very corpus it was generated from.
    """


def build_output_schema(converted_ground_truths: Iterable[dict]) -> dict:
    """Build the §6.5 output schema from converted ground truths.

    ``converted_ground_truths`` are the objects returned by
    :func:`vlm_lab.data.convert_ground_truth` for the train + validation
    splits.

    The property set at every level is exactly the keys observed at that
    path: no key is added by judgement and none is dropped for rarity.
    Every generated object node is strict (``additionalProperties: false``)
    and every key is optional (``required: []``), because the prompt
    instructs the model to omit absent keys.

    Raises:
        ValueError: if the corpus is empty, or if any element is not a JSON
            object.
        SchemaShapeError: if any path holds heterogeneous or unsupported
            shapes.
    """
    documents = list(converted_ground_truths)
    if not documents:
        raise ValueError(
            "cannot build an output schema from an empty corpus: "
            "the property set is defined as the keys observed in the corpus"
        )
    for position, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(
                f"converted ground truth at index {position} must be a JSON object, "
                f"got {_shape_name(document)}: {document!r}"
            )

    return {
        "$schema": JSON_SCHEMA_DRAFT_URI,
        **_build_object_node(documents, _ROOT_PATH),
    }


def schema_hash(schema: dict) -> str:
    """Return the SHA-256 of a canonical serialization of `schema`.

    Canonical means sorted keys and compact separators, so the hash depends
    on the schema's content and not on key insertion order. Recorded in
    result artifacts so a later regeneration cannot silently change meaning.
    """
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_against_schema(instance: object, schema: dict) -> list[str]:
    """Validate `instance` against `schema`, returning human-readable violations.

    An empty list means the instance is valid. Messages are sorted so the
    result is deterministic for a given instance and schema.
    """
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)
    return sorted(
        f"{_format_instance_path(error.absolute_path)}: {error.message}"
        for error in validator.iter_errors(instance)
    )


def _shape_name(value: object) -> str:
    """Name `value`'s JSON shape, for schema construction and error messages."""
    return _SHAPE_NAMES_BY_TYPE.get(type(value), type(value).__name__)


def _build_object_node(objects: list[dict], path: str) -> dict:
    """Build a strict object node from every object observed at `path`.

    Keys are sorted so the generated document depends only on the set of
    observed keys, not on the corpus's iteration order.
    """
    observed_keys = sorted({key for obj in objects for key in obj})
    properties = {
        key: _build_node(
            [obj[key] for obj in objects if key in obj],
            f"{path}.{key}",
        )
        for key in observed_keys
    }
    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
        "required": [],
    }


def _build_node(values: list[object], path: str) -> dict:
    """Infer the schema node for every value observed at `path`.

    All strings -> a string leaf. All objects -> a recursed object node.
    All arrays -> an array whose ``items`` is the recursed object node built
    from the elements. Anything else fails loudly.
    """
    observed_shapes = sorted({_shape_name(value) for value in values})
    unsupported_shapes = [
        shape for shape in observed_shapes if shape not in _SUPPORTED_SHAPES
    ]
    if unsupported_shapes:
        raise SchemaShapeError(
            f"unsupported value shape at path {path!r}: observed "
            f"{', '.join(observed_shapes)}; only {', '.join(_SUPPORTED_SHAPES)} "
            f"(and arrays of objects) can be encoded"
        )
    if len(observed_shapes) > 1:
        raise SchemaShapeError(
            f"heterogeneous value shapes at path {path!r}: observed "
            f"{', '.join(observed_shapes)}; a single shape is required across the corpus"
        )

    (shape,) = observed_shapes
    if shape == "string":
        return {"type": "string"}
    if shape == "object":
        return _build_object_node(values, path)
    return _build_array_node(values, path)


def _build_array_node(arrays: list[list], path: str) -> dict:
    """Build an array node from every array observed at `path`.

    An empty array carries no evidence about element shape, so a path
    observed only as empty arrays yields an unconstrained ``items``: the
    absence of evidence must not be turned into a decision.
    """
    elements = [element for array in arrays for element in array]
    if not elements:
        return {"type": "array"}

    element_shapes = sorted({_shape_name(element) for element in elements})
    if element_shapes != ["object"]:
        raise SchemaShapeError(
            f"unsupported array element shapes at path '{path}[]': observed "
            f"{', '.join(element_shapes)}; arrays must contain objects only"
        )
    return {"type": "array", "items": _build_object_node(elements, f"{path}[]")}


def _format_instance_path(path_parts: Iterable[object]) -> str:
    """Render a jsonschema error path as `$.menu[0].nm`."""
    rendered = _ROOT_PATH
    for part in path_parts:
        rendered += f"[{part}]" if isinstance(part, int) else f".{part}"
    return rendered
