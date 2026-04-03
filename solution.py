def parse_nested_parens(paren_string: str) -> List[int]:
    groups = paren_string.split()
    results = []
    for group in groups:
        max_depth = 0
        current_depth = 0
        for char in group:
            if char == '('
                current_depth += 1
                if current_depth > max_depth:
                    max_depth = current_depth
            elif char == ')'
                current_depth -= 1
        results.append(max_depth)
    return results

property_test = parse_nested_parens('(()()) ((())) () ((())()())')
assert property_test == [2, 3, 1, 3]