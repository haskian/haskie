"""How DBOS stores what a workflow is given and returns: pickle, with every `msgspec.Struct` kept
by field name.

DBOS pickles workflow inputs, outputs, step results and events into its tables, and reads them
back when it resumes a run (`workflows.adopt_orphans`) and when the Operations view lists one. A
Struct pickles by position by default: the class and its values in field order. So removing a
field fails every record written before (`Extra positional arguments provided`), and inserting
one loads each later value into the field before it, without an error. Here a Struct is stored as
its class and a dict of its fields, and rebuilt by name:

- a field removed since is dropped, a field added since takes its default, and a field moved
  keeps its value;
- a field added since without a default fails the load (`TypeError`), rather than guessing;
- the value of a removed field is still unpickled before it is dropped, so its type has to stay
  importable.

Records DBOS wrote with its own pickle are tagged `py_pickle`, and DBOS still reads those itself.
The new name is what tells the two formats apart.
"""

import base64
import io
import pickle
from typing import Any

import msgspec
from dbos import Serializer


def _rebuild[S: msgspec.Struct](cls: type[S], fields: dict[str, Any]) -> S:
    """`cls` from the fields recorded for it that it still has."""
    return cls(**{name: value for name, value in fields.items() if name in cls.__struct_fields__})


class _ByName(pickle.Pickler):
    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, msgspec.Struct):
            return _rebuild, (type(obj), msgspec.structs.asdict(obj))
        return NotImplemented


class StructSerializer(Serializer):
    def serialize(self, data: Any) -> str:
        buffer = io.BytesIO()
        _ByName(buffer).dump(data)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def deserialize(self, serialized_data: str) -> Any:
        return pickle.loads(base64.b64decode(serialized_data))

    def name(self) -> str:
        return "haskie_pickle"


SERIALIZER = StructSerializer()
