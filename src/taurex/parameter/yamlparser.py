"""Read YAML using the existing parameter-file structure and value conversion."""

import typing as t

import yaml


class _ParameterLoader(yaml.SafeLoader):
    """Leave implicit scalar conversion to ParameterParser.transform.

    In particular, molecular names such as NO must remain string keys, and
    scientific notation must behave the same way as in parameter files.
    """

    yaml_implicit_resolvers = {}

    def construct_mapping(self, node, deep=False):
        """Reject duplicate keys instead of silently discarding configuration."""
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("YAML parameter keys must be strings")
            if key in result:
                raise ValueError(f"Duplicate YAML parameter key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def read_yaml(stream: t.TextIO) -> t.Dict[str, t.Any]:
    """Read sections, subsections and scalar values from a YAML stream."""
    config = yaml.load(stream, Loader=_ParameterLoader)  # noqa: S506
    if not isinstance(config, dict):
        raise ValueError("YAML parameter file must contain a mapping of sections")
    if any(not isinstance(value, dict) for value in config.values()):
        raise ValueError("YAML top-level sections must be mappings; use {} if empty")

    def validate(section, ancestors):
        if id(section) in ancestors:
            raise ValueError("Recursive YAML aliases are not supported")
        ancestors = ancestors | {id(section)}
        for key, value in section.items():
            if isinstance(value, dict):
                validate(value, ancestors)
            else:
                values = value if isinstance(value, list) else [value]
                if any(not isinstance(v, (str, int, float, bool)) for v in values):
                    raise ValueError(
                        f"YAML parameter {key} must be a scalar or a list of scalars"
                    )

    validate(config, set())
    return config
