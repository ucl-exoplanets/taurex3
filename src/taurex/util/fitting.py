import typing as t


class MalformedPriorInputError(Exception):
    """Raised when the input prior string is malformed"""

    pass


def validate_priors(prior_string: str) -> None:
    count_left_bracket = prior_string.count("(")
    count_right_bracket = prior_string.count(")")

    if (
        count_left_bracket != count_right_bracket
        or count_left_bracket == 0
        or count_right_bracket == 0
    ):
        raise MalformedPriorInputError("Parenthesis are not balanced")

    # if count_left_bracket != 1:
    #     raise MalformedPriorInput("Only a single parenthesis pair allowed")


def parse_priors(prior_string: str) -> t.Tuple[str, t.Dict[str, t.Any]]:
    import ast

    func_parse = ast.parse(prior_string)

    actual_func = func_parse.body[0].value

    function_name = actual_func.func.id

    func_args = {kw.arg: ast.literal_eval(kw.value) for kw in actual_func.keywords}

    return function_name, func_args
