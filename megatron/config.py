# coding=utf-8
# Copyright (c) 2023, ADEPT AI LABS INC.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config objects.

Provides a bridge between the flat `args` namespace and having different
configs at any layer of the model.

Config objects can be initialized manually, but to match current `args`
behavior, they can be initialized using the `from_args` method.
"""

import argparse
import dataclasses
from typing import Type, TypeVar

T = TypeVar("T", bound="Config")

DISABLE_PREFIX = lambda: dataclasses.field(metadata={"no_from_args_prefix": True})


@dataclasses.dataclass
class Config:
    @classmethod
    def from_args(cls: Type[T], args: argparse.Namespace, arg_prefix: str = None) -> T:
        """Initialize a Config object using an `args` object.

        Args:
            args: Parsed arguments from `megatron.get_args()`. The field names
              in the Config object will be populated by `args` entries of the
              same name.
            arg_prefix: If provided, will use this prefix when finding the
              field in `args`. For example, if the field name is `hidden_size`
              and arg_prefix is `blah_`, the field will be populated by the arg
              `blah_hidden_size`.
              However, if the field has the metadata field `no_from_args_prefix`
              set to True, the prefix will not be added. This is to support
              fields that are globally applicable.
        """
        field_values = {}
        for field in dataclasses.fields(cls):
            if issubclass(field.type, Config):
                field_values[field.name] = field.type.from_args(args, arg_prefix)
            else:
                if arg_prefix:
                    prefixed_field_name = arg_prefix + field.name
                    if arg_prefix and not (
                        "no_from_args_prefix" in field.metadata and field.metadata["no_from_args_prefix"]
                    ):
                        arg_name = prefixed_field_name
                    else:
                        arg_name = field.name
                        if prefixed_field_name in vars(args):
                            raise ValueError(
                                f"{field.name} has no_from_args_prefix set, but {prefixed_field_name} exists in args. This is likely a mistake."
                            )
                else:
                    arg_name = field.name

                if arg_name not in vars(args):
                    raise ValueError(
                        f"{arg_name} not found in args when attempting to construct {cls.__name__} from args."
                    )
                field_values[field.name] = vars(args)[arg_name]
        return cls(**field_values)
