---
language:
- code
- en
license: mit
task_categories:
- text-generation
pretty_name: RepoExec
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: project
    dtype: string
  - name: module
    dtype: string
  - name: entry_point
    dtype: string
  - name: solution
    dtype: string
  - name: target_function_prompt
    dtype: string
  - name: function_signature
    dtype: string
  - name: docstring
    dtype: string
  - name: original_docstring
    dtype: string
  - name: docstring_tokens
    sequence: string
  - name: cross_context
    dtype: bool
  - name: isContained
    dtype: bool
  - name: raw_solution
    dtype: string
  - name: check
    dtype: string
  - name: test_list
    sequence: string
  - name: coverage
    dtype: float64
  - name: prompt
    dtype: string
  splits:
  - name: full_context
    num_bytes: 17679411
    num_examples: 355
  - name: medium_context
    num_bytes: 17467754
    num_examples: 355
  - name: small_context
    num_bytes: 17344466
    num_examples: 355
  download_size: 12471129
  dataset_size: 52491631
configs:
- config_name: default
  data_files:
  - split: full_context
    path: data/full_context-*
  - split: medium_context
    path: data/medium_context-*
  - split: small_context
    path: data/small_context-*
viewer: true
---



## Table of Contents
- [Dataset Description](#dataset-description)
- [Supported Tasks](#supported-tasks)
- [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Usage](#usage)
- [Additional Information](#additional-information)
- - [Other Resources](#other-resources)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)


## Dataset Description

- **Repository:** [FSoft-AI4Code/RepoExec](https://github.com/FSoft-AI4Code/RepoExec)
- **Paper:** [RepoExec: Evaluate Code Generation with a Repository-Level Executable Benchmark](https://arxiv.org/html/2406.11927v1)
- **Contact:** support.ailab@fpt.com
- **Website:** https://www.fpt-aicenter.com/ai-residency/

  
# RepoExec: Evaluate Code Generation with a Repository-Level Executable Benchmark


## Dataset Summary
RepoExec is a novel benchmark designed to evaluate code generation at the repository level with a focus on executability and correctness. This benchmark addresses the gaps in existing systems by emphasizing real-world applicability and providing a comprehensive assessment of code functionality. It aims to provide a comprehensive evaluation of code functionality and alignment with developer intent, paving the way for more reliable and applicable CodeLLMs in real-world scenarios.


## Supported Tasks
RepoExec is Repository-Level Code Generation, focus on Executability, Correctness from Test Cases and Usage of Contexts from Cross-file Dependencies. For more details and to run evaluation, please follow instruction in [RepoExec Github](https://github.com/FSoft-AI4Code/RepoExec).

## Languages
Currently, RepoExec supports Python repositories.

## Dataset Structure
### Data Instances
```
{
    "id": 0,
    "project": "test-apps/python-string-utils",
    "module": "string_utils.manipulation",
    "entry_point": "reverse",
    "solution": "def reverse(input_string: str) -> str:\n    \"\"\"\n    Returns the string with its chars reversed.\n\n    *Example:*\n\n    >>> reverse('hello') # returns 'olleh'\n\n    :param input_string: String to revert.\n    :type input_string: str\n    :return: Reversed string.\n    \"\"\"\n    if not is_string(input_string):\n        raise InvalidInputError(input_string)\n\n    return input_string[::-1]",
    "prompt": "import base64\nimport random\nimport unicodedata\nimport zlib\nfrom typing import Union\nfrom uuid import uuid4\nfrom ._regex import *\nfrom .errors import InvalidInputError\nfrom .validation import is_snake_case, is_full_string, is_camel_case, is_integer, is_string\n\nclass InvalidInputError(TypeError):\n    \"\"\"\n    Custom error raised when received object is not a string as expected.\n    \"\"\"\n\n    def __init__(self, input_data: Any):\n        \"\"\"\n        :param input_data: Any received object\n        \"\"\"\n        type_name = type(input_data).__name__\n        msg = 'Expected \"str\", received \"{}\"'.format(type_name)\n        super().__init__(msg)\n\ndef is_string(obj: Any) -> bool:\n    \"\"\"\n    Checks if an object is a string.\n\n    *Example:*\n\n    >>> is_string('foo') # returns true\n    >>> is_string(b'foo') # returns false\n\n    :param obj: Object to test.\n    :return: True if string, false otherwise.\n    \"\"\"\n    return isinstance(obj, str)\n\ndef reverse(input_string: str) -> str:\n    \"\"\"\n    Returns the string with its chars reversed.\n\n    *Example:*\n\n    >>> reverse('hello') # returns 'olleh'\n\n    :param input_string: String to revert.\n    :type input_string: str\n    :return: Reversed string.\n    \"\"\"\n",
    "target_function_prompt": "def reverse(input_string: str) -> str:\n    \"\"\"\n    Returns the string with its chars reversed.\n\n    *Example:*\n\n    >>> reverse('hello') # returns 'olleh'\n\n    :param input_string: String to revert.\n    :type input_string: str\n    :return: Reversed string.\n    \"\"\"\n",
    "function_signature": "def reverse(input_string: str) -> str:",
    "docstring": "\nReturns the string with its chars reversed.\n\n*Example:*\n\n>>> reverse('hello') # returns 'olleh'\n\n:param input_string: String to revert.\n:type input_string: str\n:return: Reversed string.\n",
    "original_docstring": "\"\"\"\nReturns the string with its chars reversed.\n\n*Example:*\n\n>>> reverse('hello') # returns 'olleh'\n\n:param input_string: String to revert.\n:type input_string: str\n:return: Reversed string.\n\"\"\"",
    "docstring_tokens": [
        "Returns",
        "the",
        "string",
        "with",
        "its",
        "chars",
        "reversed",
        ".",
        "*",
        "Example",
        ":",
        "*",
        ">>>",
        "reverse",
        "(",
        "'",
        "hello",
        "'",
        ")",
        "#",
        "returns",
        "'",
        "olleh",
        "'",
        ":",
        "param",
        "input_string",
        ":",
        "String",
        "to",
        "revert",
        ".",
        ":",
        "type",
        "input_string",
        ":",
        "str",
        ":",
        "return",
        ":",
        "Reversed",
        "string",
        "."
    ],
    "cross_context": true,
    "isContained": false,
    "raw_solution": "def reverse(input_string: str) -> str:\n    \"\"\"\n    Returns the string with its chars reversed.\n\n    *Example:*\n\n    >>> reverse('hello') # returns 'olleh'\n\n    :param input_string: String to revert.\n    :type input_string: str\n    :return: Reversed string.\n    \"\"\"\n    if not is_string(input_string):\n        raise InvalidInputError(input_string)\n\n    return input_string[::-1]",
    "check": "\nimport sys\nsys.path.insert(1, \"/input/test-apps/python-string-utils\")\nimport unittest, pytest\nimport math\nimport random\nimport re\nimport copy\nimport datetime\nimport itertools\nimport collections\nimport heapq\nimport statistics\nimport functools\nimport hashlib\nimport numpy\nimport numpy as np\nimport string\nfrom typing import *\nfrom collections import *\nimport pickle\nimport timeout_decorator\n\n\n__all__ = [\n    'camel_case_to_snake',\n    'snake_case_to_camel',\n    'reverse',\n    'shuffle',\n    'strip_html',\n    'prettify',\n    'asciify',\n    'slugify',\n    'booleanize',\n    'strip_margin',\n    'compress',\n    'decompress',\n    'roman_encode',\n    'roman_decode',\n]\n\nimport base64\nimport random\nimport unicodedata\nimport zlib\nfrom typing import Union\nfrom uuid import uuid4\n\nfrom string_utils._regex import *\nfrom string_utils.errors import InvalidInputError\nfrom string_utils.validation import is_snake_case, is_full_string, is_camel_case, is_integer, is_string\n\n\n\n\nclass __RomanNumbers:\n    # internal rule mappings for encode()\n    __mappings = [\n        # units\n        {1: 'I', 5: 'V'},\n        # tens\n        {1: 'X', 5: 'L'},\n        # hundreds\n        {1: 'C', 5: 'D'},\n        # thousands\n        {1: 'M'},\n    ]\n\n    # swap key/value definitions for decode()\n    __reversed_mappings = [{v: k for k, v in m.items()} for m in __mappings]\n\n    @classmethod\n    def __encode_digit(cls, index: int, value: int) -> str:\n        # if digit is zero, there is no sign to display\n        if value == 0:\n            return ''\n\n        # from 1 to 3 we have just to repeat the sign N times (eg: III, XXX...)\n        if value <= 3:\n            return cls.__mappings[index][1] * value\n\n        # if 4 we have to add unit prefix\n        if value == 4:\n            return cls.__mappings[index][1] + cls.__mappings[index][5]\n\n        # if is 5, is a straight map\n        if value == 5:\n            return cls.__mappings[index][5]\n\n        # if 6, 7 or 8 we have to append unit suffixes\n        if value <= 8:\n            suffix = cls.__mappings[index][1] * (value - 5)\n            return cls.__mappings[index][5] + suffix\n\n        # if 9 we have to prepend current unit to next\n        return cls.__mappings[index][1] + cls.__mappings[index + 1][1]\n\n    @classmethod\n    def encode(cls, input_number: Union[str, int]) -> str:\n        # force input conversion to a string (we need it in order to iterate on each digit)\n        input_string = str(input_number)\n\n        if not is_integer(input_string):\n            raise ValueError('Invalid input, only strings or integers are allowed')\n\n        value = int(input_string)\n\n        if value < 1 or value > 3999:\n            raise ValueError('Input must be >= 1 and <= 3999')\n\n        input_len = len(input_string)\n        output = ''\n\n        # decode digits from right to left (start from units to thousands)\n        for index in range(input_len):\n            # get actual digit value as int\n            digit = int(input_string[input_len - index - 1])\n\n            # encode digit to roman string\n            encoded_digit = cls.__encode_digit(index, digit)\n\n            # prepend encoded value to the current output in order to have the final string sorted\n            # from thousands to units\n            output = encoded_digit + output\n\n        return output\n\n    @classmethod\n    def __index_for_sign(cls, sign: str) -> int:\n        for index, mapping in enumerate(cls.__reversed_mappings):\n            if sign in mapping:\n                return index\n\n        raise ValueError('Invalid token found: \"{}\"'.format(sign))\n\n    @classmethod\n    def decode(cls, input_string: str) -> int:\n        if not is_full_string(input_string):\n            raise ValueError('Input must be a non empty string')\n\n        # reverse the provided string so that we can start parsing from units to thousands\n        reversed_string = reverse(input_string.upper())\n\n        # track last used value\n        last_value = None\n\n        # computed number to return\n        output = 0\n\n        # for each sign in the string we get its numeric value and add or subtract it to the computed output\n        for sign in reversed_string:\n            # are we dealing with units, tens, hundreds or thousands?\n            index = cls.__index_for_sign(sign)\n\n            # it's basically 1 or 5 (based on mapping rules definitions)\n            key_value = cls.__reversed_mappings[index][sign]\n\n            # Based on the level (tens, hundreds...) we have to add as many zeroes as the level into which we are\n            # in order to have the actual sign value.\n            # For instance, if we are at level 2 we are dealing with hundreds, therefore instead of 1 or 5, we will\n            # obtain 100 or 500 by adding 2 zeroes\n            sign_value = int(str(key_value) + '0' * index)\n\n            # increase total value if we are moving on with level\n            if last_value is None or sign_value >= last_value:\n                output += sign_value\n\n            # Decrease value if we are back to a previous level\n            # For instance, if we are parsing \"IX\", we first encounter \"X\" which is ten then \"I\" which is unit,\n            # So we have to do the following operation in order to get 9 (the final result): 10 - 1\n            else:\n                output -= sign_value\n\n            last_value = sign_value\n\n        return output\n\n\nclass __StringCompressor:\n\n    @staticmethod\n    def __require_valid_input_and_encoding(input_string: str, encoding: str):\n        if not is_string(input_string):\n            raise InvalidInputError(input_string)\n\n        if len(input_string) == 0:\n            raise ValueError('Input string cannot be empty')\n\n        if not is_string(encoding):\n            raise ValueError('Invalid encoding')\n\n    @classmethod\n    def compress(cls, input_string: str, encoding: str = 'utf-8', compression_level: int = 9) -> str:\n        cls.__require_valid_input_and_encoding(input_string, encoding)\n\n        if not isinstance(compression_level, int) or compression_level < 0 or compression_level > 9:\n            raise ValueError('Invalid compression_level: it must be an \"int\" between 0 and 9')\n\n        # turns input string into a sequence of bytes using provided encoding\n        original_bytes = input_string.encode(encoding)\n\n        # compress bytes using zlib library\n        compressed_bytes = zlib.compress(original_bytes, compression_level)\n\n        # encode compressed bytes using base64\n        # (this ensure that all characters will be available and that the output string can be used safely in any\n        # context such URLs)\n        encoded_bytes = base64.urlsafe_b64encode(compressed_bytes)\n\n        # finally turns base64 bytes into a string\n        output = encoded_bytes.decode(encoding)\n\n        return output\n\n    @classmethod\n    def decompress(cls, input_string: str, encoding: str = 'utf-8') -> str:\n        cls.__require_valid_input_and_encoding(input_string, encoding)\n\n        # turns input string into a sequence of bytes\n        # (the string is assumed to be a previously compressed string, therefore we have to decode it using base64)\n        input_bytes = base64.urlsafe_b64decode(input_string)\n\n        # decompress bytes using zlib\n        decompressed_bytes = zlib.decompress(input_bytes)\n\n        # decode the decompressed bytes to get the original string back\n        original_string = decompressed_bytes.decode(encoding)\n\n        return original_string\n\n\nclass __StringFormatter:\n    def __init__(self, input_string):\n        if not is_string(input_string):\n            raise InvalidInputError(input_string)\n\n        self.input_string = input_string\n\n    def __uppercase_first_char(self, regex_match):\n        return regex_match.group(0).upper()\n\n    def __remove_duplicates(self, regex_match):\n        return regex_match.group(1)[0]\n\n    def __uppercase_first_letter_after_sign(self, regex_match):\n        match = regex_match.group(1)\n        return match[:-1] + match[2].upper()\n\n    def __ensure_right_space_only(self, regex_match):\n        return regex_match.group(1).strip() + ' '\n\n    def __ensure_left_space_only(self, regex_match):\n        return ' ' + regex_match.group(1).strip()\n\n    def __ensure_spaces_around(self, regex_match):\n        return ' ' + regex_match.group(1).strip() + ' '\n\n    def __remove_internal_spaces(self, regex_match):\n        return regex_match.group(1).strip()\n\n    def __fix_saxon_genitive(self, regex_match):\n        return regex_match.group(1).replace(' ', '') + ' '\n\n    # generates a placeholder to inject temporary into the string, it will be replaced with the original\n    # value at the end of the process\n    @staticmethod\n    def __placeholder_key():\n        return '$' + uuid4().hex + '$'\n\n    def format(self) -> str:\n        # map of temporary placeholders\n        placeholders = {}\n        out = self.input_string\n\n        # looks for url or email and updates placeholders map with found values\n        placeholders.update({self.__placeholder_key(): m[0] for m in URLS_RE.findall(out)})\n        placeholders.update({self.__placeholder_key(): m for m in EMAILS_RE.findall(out)})\n\n        # replace original value with the placeholder key\n        for p in placeholders:\n            out = out.replace(placeholders[p], p, 1)\n\n        out = PRETTIFY_RE['UPPERCASE_FIRST_LETTER'].sub(self.__uppercase_first_char, out)\n        out = PRETTIFY_RE['DUPLICATES'].sub(self.__remove_duplicates, out)\n        out = PRETTIFY_RE['RIGHT_SPACE'].sub(self.__ensure_right_space_only, out)\n        out = PRETTIFY_RE['LEFT_SPACE'].sub(self.__ensure_left_space_only, out)\n        out = PRETTIFY_RE['SPACES_AROUND'].sub(self.__ensure_spaces_around, out)\n        out = PRETTIFY_RE['SPACES_INSIDE'].sub(self.__remove_internal_spaces, out)\n        out = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].sub(self.__uppercase_first_letter_after_sign, out)\n        out = PRETTIFY_RE['SAXON_GENITIVE'].sub(self.__fix_saxon_genitive, out)\n        out = out.strip()\n\n        # restore placeholder keys with their associated original value\n        for p in placeholders:\n            out = out.replace(p, placeholders[p], 1)\n\n        return out\n\n\n\ndef reverse(input_string: str) -> str:\n    \"\"\"\n    Returns the string with its chars reversed.\n\n    *Example:*\n\n    >>> reverse('hello') # returns 'olleh'\n\n    :param input_string: String to revert.\n    :type input_string: str\n    :return: Reversed string.\n    \"\"\"\n    if not is_string(input_string):\n        raise InvalidInputError(input_string)\n\n    return input_string[::-1]\n\n\ndef camel_case_to_snake(input_string, separator='_'):\n    \"\"\"\n    Convert a camel case string into a snake case one.\n    (The original string is returned if is not a valid camel case string)\n\n    *Example:*\n\n    >>> camel_case_to_snake('ThisIsACamelStringTest') # returns 'this_is_a_camel_case_string_test'\n\n    :param input_string: String to convert.\n    :type input_string: str\n    :param separator: Sign to use as separator.\n    :type separator: str\n    :return: Converted string.\n    \"\"\"\n    if not is_string(input_string):\n        raise InvalidInputError(input_string)\n\n    if not is_camel_case(input_string):\n        return input_string\n\n    return CAMEL_CASE_REPLACE_RE.sub(lambda m: m.group(1) + separator, input_string).lower()\n\n\ndef snake_case_to_camel(input_string: str, upper_case_first: bool = True, separator: str = '_') -> str:\n    \"\"\"\n    Convert a snake case string into a camel case one.\n    (The original string is returned if is not a valid snake case string)\n\n    *Example:*\n\n    >>> snake_case_to_camel('the_snake_is_green') # returns 'TheSnakeIsGreen'\n\n    :param input_string: String to convert.\n    :type input_string: str\n    :param upper_case_first: True to turn the first letter into uppercase (default).\n    :type upper_case_first: bool\n    :param separator: Sign to use as separator (default to \"_\").\n    :type separator: str\n    :return: Converted string\n    \"\"\"\n    if not is_string(input_string):\n        raise InvalidInputError(input_string)\n\n    if not is_snake_case(input_string, separator):\n        return input_string\n\n    tokens = [s.title() for s in input_string.split(separator) if is_full_string(s)]\n\n    if not upper_case_first:\n        tokens[0] = tokens[0].lower()\n\n    out = ''.join(tokens)\n\n    return out\n\n\ndef shuffle(input_string: str) -> str:\n    \"\"\"\n    Return a new string containing same chars of the given one but in a randomized order.\n\n    *Example:*\n\n    >>> shuffle('hello world') # possible output: 'l wodheorll'\n\n    :param input_string: String to shuffle\n    :type input_string: str\n    :return: Shuffled string\n    \"\"\"\n    if not is_string(input_string):\n        raise InvalidInputError(input_string)\n\n    # turn the string into a list of chars\n    chars = list(input_string)\n\n    # shuffle the list\n    random.shuffle(chars)\n\n    # convert the shuffled list back to string\n    return ''.join(chars)\n\n\ndef strip_html(input_string: str, keep_tag_content: bool = False) -> str:\n    \"\"\"\n    Remove html code contained into the given string.\n\n    *Examples:*\n\n    >>> strip_html('test: <a href=\"foo/bar\">click here</a>') # returns 'test: '\n    >>> strip_html('test: <a href=\"foo/bar\">click here</a>', keep_tag_content=True) # returns 'test: click here'\n\n    :param input_string: String to manipulate.\n    :type input_string: str\n    :param keep_tag_content: True to preserve tag content, False to remove tag and its content too (default).\n    :type keep_tag_content: bool\n    :return: String with html removed.\n    \"\"\"\n    if not is_string(input_string):\n        raise InvalidInputError(input_string)\n\n    r = HTML_TAG_ONLY_RE if keep_tag_content else HTML_RE\n\n    return r.sub('', input_string)\n\n\ndef prettify(input_string: str) -> str:\n    \"\"\"\n    Reformat a string by applying the following basic grammar and formatting rules:\n\n    - String cannot start or end with spaces\n    - The first letter in the string and the ones after a dot, an exclamation or a question mark must be uppercase\n    - String cannot have multiple sequential spaces, empty lines or punctuation (except for \"?\", \"!\" and \".\")\n    - Arithmetic operators (+, -, /, \\\\*, =) must have one, and only one space before and after themselves\n    - One, and only one space should follow a dot, a comma, an exclamation or a question mark\n    - Text inside double quotes cannot start or end with spaces, but one, and only one space must come first and \\\n    after quotes (foo\" bar\"baz -> foo \"bar\" baz)\n    - Text inside round brackets cannot start or end with spaces, but one, and only one space must come first and \\\n    after brackets (\"foo(bar )baz\" -> \"foo (bar) baz\")\n    - Percentage sign (\"%\") cannot be preceded by a space if there is a number before (\"100 %\" -> \"100%\")\n    - Saxon genitive is correct (\"Dave' s dog\" -> \"Dave's dog\")\n\n    *Examples:*\n\n    >>> prettify(' unprettified string ,, like this one,will be\"prettified\" .it\\\\' s awesome! ')\n    >>> # -> 'Unprettified string, like this one, will be \"prettified\". It\\'s awesome!'\n\n    :param input_string: String to manipulate\n    :return: Prettified string.\n    \"\"\"\n    formatted = __StringFormatter(input_string).format()\n    return formatted\n\n\ndef asciify(input_string: str) -> str:\n    \"\"\"\n    Force string content to be ascii-only by translating all non-ascii chars into the closest possible representation\n    (eg: \u00f3 -> o, \u00cb -> E, \u00e7 -> c...).\n\n    **Bear in mind**: Some chars may be lost if impossible to translate.\n\n    *Example:*\n\n    >>> asciify('\u00e8\u00e9\u00f9\u00fa\u00f2\u00f3\u00e4\u00e5\u00eb\u00fd\u00f1\u00c5\u00c0\u00c1\u00c7\u00cc\u00cd\u00d1\u00d3\u00cb') # returns 'eeuuooaaeynAAACIINOE'\n\n    :param input_string: String to convert\n    :return: Ascii utf-8 string\n    \"\"\"\n    if not is_string(input_string):\n        raise InvalidInputError(input_string)\n\n    # \"NFKD\" is the algorithm which is able to successfully translate the most of non-ascii chars\n    normalized = unicodedata.normalize('NFKD', input_string)\n\n    # encode string forcing ascii and ignore any errors (unrepresentable chars will be stripped out)\n    ascii_bytes = normalized.encode('ascii', 'ignore')\n\n    # turns encoded bytes into an utf-8 string\n    ascii_string = ascii_bytes.decode('utf-8')\n\n    return ascii_string\n\n\ndef slugify(input_string: str, separator: str = '-') -> str:\n    \"\"\"\n    Converts a string into a \"slug\" using provided separator.\n    The returned string has the following properties:\n\n    - it has no spaces\n    - all letters are in lower case\n    - all punctuation signs and non alphanumeric chars are removed\n    - words are divided using provided separator\n    - all chars are encoded as ascii (by using `asciify()`)\n    - is safe for URL\n\n    *Examples:*\n\n    >>> slugify('Top 10 Reasons To Love Dogs!!!') # returns: 'top-10-reasons-to-love-dogs'\n    >>> slugify('M\u00f6nst\u00e9r M\u00e4gn\u00ebt') # returns 'monster-magnet'\n\n    :param input_string: String to convert.\n    :type input_string: str\n    :param separator: Sign used to join string tokens (default to \"-\").\n    :type separator: str\n    :return: Slug string\n    \"\"\"\n    if not is_string(input_string):\n        raise InvalidInputError(input_string)\n\n    # replace any character that is NOT letter or number with spaces\n    out = NO_LETTERS_OR_NUMBERS_RE.sub(' ', input_string.lower()).strip()\n\n    # replace spaces with join sign\n    out = SPACES_RE.sub(separator, out)\n\n    # normalize joins (remove duplicates)\n    out = re.sub(re.escape(separator) + r'+', separator, out)\n\n    return asciify(out)\n\n\ndef booleanize(input_string: str) -> bool:\n    \"\"\"\n    Turns a string into a boolean based on its content (CASE INSENSITIVE).\n\n    A positive boolean (True) is returned if the string value is one of the following:\n\n    - \"true\"\n    - \"1\"\n    - \"yes\"\n    - \"y\"\n\n    Otherwise False is returned.\n\n    *Examples:*\n\n    >>> booleanize('true') # returns True\n    >>> booleanize('YES') # returns True\n    >>> booleanize('nope') # returns False\n\n    :param input_string: String to convert\n    :type input_string: str\n    :return: True if the string contains a boolean-like positive value, false otherwise\n    \"\"\"\n    if not is_string(input_string):\n        raise InvalidInputError(input_string)\n\n    return input_string.lower() in ('true', '1', 'yes', 'y')\n\n\ndef strip_margin(input_string: str) -> str:\n    \"\"\"\n    Removes tab indentation from multi line strings (inspired by analogous Scala function).\n\n    *Example:*\n\n    >>> strip_margin('''\n    >>>                 line 1\n    >>>                 line 2\n    >>>                 line 3\n    >>> ''')\n    >>> # returns:\n    >>> '''\n    >>> line 1\n    >>> line 2\n    >>> line 3\n    >>> '''\n\n    :param input_string: String to format\n    :type input_string: str\n    :return: A string without left margins\n    \"\"\"\n    if not is_string(input_string):\n        raise InvalidInputError(input_string)\n\n    line_separator = '\\n'\n    lines = [MARGIN_RE.sub('', line) for line in input_string.split(line_separator)]\n    out = line_separator.join(lines)\n\n    return out\n\n\ndef compress(input_string: str, encoding: str = 'utf-8', compression_level: int = 9) -> str:\n    \"\"\"\n    Compress the given string by returning a shorter one that can be safely used in any context (like URL) and\n    restored back to its original state using `decompress()`.\n\n    **Bear in mind:**\n    Besides the provided `compression_level`, the compression result (how much the string is actually compressed\n    by resulting into a shorter string) depends on 2 factors:\n\n    1. The amount of data (string size): short strings might not provide a significant compression result\\\n    or even be longer than the given input string (this is due to the fact that some bytes have to be embedded\\\n    into the compressed string in order to be able to restore it later on)\\\n\n    2. The content type: random sequences of chars are very unlikely to be successfully compressed, while the best\\\n    compression result is obtained when the string contains several recurring char sequences (like in the example).\n\n    Behind the scenes this method makes use of the standard Python's zlib and base64 libraries.\n\n    *Examples:*\n\n    >>> n = 0 # <- ignore this, it's a fix for Pycharm (not fixable using ignore comments)\n    >>> # \"original\" will be a string with 169 chars:\n    >>> original = ' '.join(['word n{}'.format(n) for n in range(20)])\n    >>> # \"compressed\" will be a string of 88 chars\n    >>> compressed = compress(original)\n\n    :param input_string: String to compress (must be not empty or a ValueError will be raised).\n    :type input_string: str\n    :param encoding: String encoding (default to \"utf-8\").\n    :type encoding: str\n    :param compression_level: A value between 0 (no compression) and 9 (best compression), default to 9.\n    :type compression_level: int\n    :return: Compressed string.\n    \"\"\"\n    return __StringCompressor.compress(input_string, encoding, compression_level)\n\n\ndef decompress(input_string: str, encoding: str = 'utf-8') -> str:\n    \"\"\"\n    Restore a previously compressed string (obtained using `compress()`) back to its original state.\n\n    :param input_string: String to restore.\n    :type input_string: str\n    :param encoding: Original string encoding.\n    :type encoding: str\n    :return: Decompressed string.\n    \"\"\"\n    return __StringCompressor.decompress(input_string, encoding)\n\n\ndef roman_encode(input_number: Union[str, int]) -> str:\n    \"\"\"\n    Convert the given number/string into a roman number.\n\n    The passed input must represents a positive integer in the range 1-3999 (inclusive).\n\n    Why this limit? You may be wondering:\n\n    1. zero is forbidden since there is no related representation in roman numbers\n    2. the upper bound 3999 is due to the limitation in the ascii charset\\\n    (the higher quantity sign displayable in ascii is \"M\" which is equal to 1000, therefore based on\\\n    roman numbers rules we can use 3 times M to reach 3000 but we can't go any further in thousands without\\\n    special \"boxed chars\").\n\n    *Examples:*\n\n    >>> roman_encode(37) # returns 'XXXVIII'\n    >>> roman_encode('2020') # returns 'MMXX'\n\n    :param input_number: An integer or a string to be converted.\n    :type input_number: Union[str, int]\n    :return: Roman number string.\n    \"\"\"\n    return __RomanNumbers.encode(input_number)\n\n\ndef roman_decode(input_string: str) -> int:\n    \"\"\"\n    Decode a roman number string into an integer if the provided string is valid.\n\n    *Example:*\n\n    >>> roman_decode('VII') # returns 7\n\n    :param input_string: (Assumed) Roman number\n    :type input_string: str\n    :return: Integer value\n    \"\"\"\n    return __RomanNumbers.decode(input_string)\n\n\nimport pickle\ndef test_0():\n    assert reverse(\"mystring\") == \"gnirtsym\"\ntest_0()\n\ndef test_1():\n    assert reverse('a') == 'a'\ntest_1()\n\ndef test_2():\n    assert reverse('hello') == 'olleh'\ntest_2()\n\ndef test_3():\n    assert reverse('hello world') == 'dlrow olleh'\ntest_3()\n\ndef test_4():\n    assert reverse(\"hello\") == \"olleh\"\ntest_4()\n\ndef test_5():\n    assert reverse('h') == 'h'\ntest_5()\n\ndef test_6():\n    assert reverse('') == ''\ntest_6()\n\ndef test_7():\n    assert reverse(\"\ud83d\ude00\") == \"\ud83d\ude00\"\ntest_7()\n\ndef test_8():\n    assert reverse('abc') == 'cba'\ntest_8()\n\ndef test_9():\n    assert reverse(\"pizza\") == \"azzip\"\ntest_9()\n\ndef test_11():\n    assert is_string(reverse('hello'))\ntest_11()\n\ndef test_14():\n    assert reverse('H') == 'H'\ntest_14()\n\ndef test_15():\n    assert reverse('bar') == 'rab'\ntest_15()\n\ndef test_16():\n    assert reverse(\"AbCdEfG\") == \"GfEdCbA\"\ntest_16()\n\ndef test_18():\n    assert \"olleh\" == reverse('hello')\ntest_18()\n\ndef test_19():\n    assert reverse('ab') == 'ba'\ntest_19()\n\ndef test_20():\n    assert reverse('Hello') == 'olleH'\ntest_20()\n\ndef test_21():\n    assert reverse('Hello, World!') == '!dlroW ,olleH'\ntest_21()\n\ndef test_22():\n    assert reverse(reverse(\"hello\")) == \"hello\"\ntest_22()\n\ndef test_23():\n    assert reverse('Hello World!') == '!dlroW olleH'\ntest_23()\n\ndef test_24():\n    assert reverse(\"world\") == \"dlrow\"\ntest_24()\n\ndef test_25():\n    assert reverse('world') == 'dlrow'\ntest_25()\n\ndef test_26():\n    assert reverse('lol') == 'lol'\ntest_26()\n\ndef test_29():\n    assert reverse('foo') == 'oof'\ntest_29()\n\ndef test_30():\n    assert reverse(reverse('hello')) == 'hello'\ntest_30()\n\ndef test_31():\n    assert reverse('o') == 'o'\ntest_31()\n\ndef test_10():\n    with open(\"/output/test-apps+python-string-utils/test_output/string_utils+manipulation/reverse/test_10\", \"rb\") as f:\n        output = pickle.load(f)\n    assert reverse('sup?') == output\ntest_10()\n\ndef test_12():\n    with open(\"/output/test-apps+python-string-utils/test_output/string_utils+manipulation/reverse/test_12\", \"rb\") as f:\n        output = pickle.load(f)\n    assert reverse(\"A0B$C\") == output\ntest_12()\n\ndef test_13():\n    with open(\"/output/test-apps+python-string-utils/test_output/string_utils+manipulation/reverse/test_13\", \"rb\") as f:\n        output = pickle.load(f)\n    assert reverse(\"{test}\") == output\ntest_13()\n\ndef test_17():\n    with open(\"/output/test-apps+python-string-utils/test_output/string_utils+manipulation/reverse/test_17\", \"rb\") as f:\n        output = pickle.load(f)\n    assert reverse(\"\u13be\u13cd\u13a9\u13be\") == output\ntest_17()\n\ndef test_27():\n    with open(\"/output/test-apps+python-string-utils/test_output/string_utils+manipulation/reverse/test_27\", \"rb\") as f:\n        output = pickle.load(f)\n    assert reverse(\"?a123\") == output\ntest_27()\n\ndef test_28():\n    with open(\"/output/test-apps+python-string-utils/test_output/string_utils+manipulation/reverse/test_28\", \"rb\") as f:\n        output = pickle.load(f)\n    assert reverse(\"software\") == output\ntest_28()\n\n\ndef test_extra_0():\n    # Test basic input\n    assert reverse(\"hello\") == \"olleh\"\n    assert reverse(\"python\") == \"nohtyp\"\n    assert reverse(\"\") == \"\"\n\n    # Test non-ASCII input\n    assert reverse(\"\ud83d\ude00\") == \"\ud83d\ude00\"\n    assert reverse(\"\u00e9\u00e7\u00e0\") == \"\u00e0\u00e7\u00e9\"\n\n    # Test input with spaces\n    assert reverse(\"hello world\") == \"dlrow olleh\"\n    assert reverse(\"  \") == \"  \"\n\n    # Test input with special characters\n    assert reverse(\"!@#$%^&*()\") == \")(*&^%$#@!\"\n    assert reverse(\"hello!@#$%^&*()\") == \")(*&^%$#@!olleh\"\n\n    # Test input with newline characters\n    assert reverse(\"hello\\nworld\") == \"dlrow\\nolleh\"\n    assert reverse(\"\\n\") == \"\\n\"\n\n    # Test input with tabs\n    assert reverse(\"\\t\\thello\") == \"olleh\\t\\t\"\n    assert reverse(\"\\t\") == \"\\t\"\n\n    # Test input with mixed types\n    with pytest.raises(InvalidInputError):\n        reverse(1234)\n\n    with pytest.raises(InvalidInputError):\n        reverse(True)\n\n    with pytest.raises(InvalidInputError):\n        reverse([1, 2, 3])\n\n    # Test performance\n    assert reverse(\"a\" * 100000) == \"a\" * 100000\n    assert reverse(\"a\" * 1000000) == \"a\" * 1000000\ntest_extra_0()\n\ndef test_extra_1():\n    try:\n        reverse(123)\n    except InvalidInputError:\n        assert True\n    else:\n        assert False\ntest_extra_1()",
    "test_list": [
        "def test_0():\n    assert reverse(\"mystring\") == \"gnirtsym\"",
        "def test_1():\n    assert reverse('a') == 'a'",
        "def test_2():\n    assert reverse('hello') == 'olleh'",
        "def test_3():\n    assert reverse('hello world') == 'dlrow olleh'",
        "def test_4():\n    assert reverse(\"hello\") == \"olleh\"",
        "def test_5():\n    assert reverse('h') == 'h'",
        "def test_6():\n    assert reverse('') == ''",
        "def test_7():\n    assert reverse(\"\ud83d\ude00\") == \"\ud83d\ude00\"",
        "def test_8():\n    assert reverse('abc') == 'cba'",
        "def test_9():\n    assert reverse(\"pizza\") == \"azzip\"",
        "def test_11():\n    assert is_string(reverse('hello'))",
        "def test_14():\n    assert reverse('H') == 'H'",
        "def test_15():\n    assert reverse('bar') == 'rab'",
        "def test_16():\n    assert reverse(\"AbCdEfG\") == \"GfEdCbA\"",
        "def test_18():\n    assert \"olleh\" == reverse('hello')",
        "def test_19():\n    assert reverse('ab') == 'ba'",
        "def test_20():\n    assert reverse('Hello') == 'olleH'",
        "def test_21():\n    assert reverse('Hello, World!') == '!dlroW ,olleH'",
        "def test_22():\n    assert reverse(reverse(\"hello\")) == \"hello\"",
        "def test_23():\n    assert reverse('Hello World!') == '!dlroW olleH'",
        "def test_24():\n    assert reverse(\"world\") == \"dlrow\"",
        "def test_25():\n    assert reverse('world') == 'dlrow'",
        "def test_26():\n    assert reverse('lol') == 'lol'",
        "def test_29():\n    assert reverse('foo') == 'oof'",
        "def test_30():\n    assert reverse(reverse('hello')) == 'hello'",
        "def test_31():\n    assert reverse('o') == 'o'",
        "def test_10():\n    with open(\"/output/test-apps+python-string-utils/test_output/string_utils+manipulation/reverse/test_10\", \"rb\") as f:\n        output = pickle.load(f)\n    assert reverse('sup?') == output",
        "def test_12():\n    with open(\"/output/test-apps+python-string-utils/test_output/string_utils+manipulation/reverse/test_12\", \"rb\") as f:\n        output = pickle.load(f)\n    assert reverse(\"A0B$C\") == output",
        "def test_13():\n    with open(\"/output/test-apps+python-string-utils/test_output/string_utils+manipulation/reverse/test_13\", \"rb\") as f:\n        output = pickle.load(f)\n    assert reverse(\"{test}\") == output",
        "def test_17():\n    with open(\"/output/test-apps+python-string-utils/test_output/string_utils+manipulation/reverse/test_17\", \"rb\") as f:\n        output = pickle.load(f)\n    assert reverse(\"\u13be\u13cd\u13a9\u13be\") == output",
        "def test_27():\n    with open(\"/output/test-apps+python-string-utils/test_output/string_utils+manipulation/reverse/test_27\", \"rb\") as f:\n        output = pickle.load(f)\n    assert reverse(\"?a123\") == output",
        "def test_28():\n    with open(\"/output/test-apps+python-string-utils/test_output/string_utils+manipulation/reverse/test_28\", \"rb\") as f:\n        output = pickle.load(f)\n    assert reverse(\"software\") == output",
        "def test_extra_0():\n    # Test basic input\n    assert reverse(\"hello\") == \"olleh\"\n    assert reverse(\"python\") == \"nohtyp\"\n    assert reverse(\"\") == \"\"\n\n    # Test non-ASCII input\n    assert reverse(\"\ud83d\ude00\") == \"\ud83d\ude00\"\n    assert reverse(\"\u00e9\u00e7\u00e0\") == \"\u00e0\u00e7\u00e9\"\n\n    # Test input with spaces\n    assert reverse(\"hello world\") == \"dlrow olleh\"\n    assert reverse(\"  \") == \"  \"\n\n    # Test input with special characters\n    assert reverse(\"!@#$%^&*()\") == \")(*&^%$#@!\"\n    assert reverse(\"hello!@#$%^&*()\") == \")(*&^%$#@!olleh\"\n\n    # Test input with newline characters\n    assert reverse(\"hello\\nworld\") == \"dlrow\\nolleh\"\n    assert reverse(\"\\n\") == \"\\n\"\n\n    # Test input with tabs\n    assert reverse(\"\\t\\thello\") == \"olleh\\t\\t\"\n    assert reverse(\"\\t\") == \"\\t\"\n\n    # Test input with mixed types\n    with pytest.raises(InvalidInputError):\n        reverse(1234)\n\n    with pytest.raises(InvalidInputError):\n        reverse(True)\n\n    with pytest.raises(InvalidInputError):\n        reverse([1, 2, 3])\n\n    # Test performance\n    assert reverse(\"a\" * 100000) == \"a\" * 100000\n    assert reverse(\"a\" * 1000000) == \"a\" * 1000000",
        "def test_extra_1():\n    try:\n        reverse(123)\n    except InvalidInputError:\n        assert True\n    else:\n        assert False"
    ],
    "coverage": 100.0
}
```
### Data Fields

Data fields for inline level:
- **id** (string): the unique id of a problem
- **project** (string): project name used to extract the data instance
- **module** (string): the module name (file) in the project used to extract the data instance
- **entry_point** (string): target function name
- **solution** (string): the gold solution of the problem
- **prompt** (string): input prompt to LLMs
- **target_function_prompt** (string): target function signature and docstring
- **function_signature** (string): target function signature
- **docstring** (string): cleaned docstring
- **original_docstring** (string): raw docstring
- **docstring_tokens** (list): list of docstring tokens,
- **cross_context** (bool): dependencies from cross file or not
- **isContained** (bool): end function or function is called by other functions
- **check** (string): code used to test the generated function
- **test_list** (list): list of unit tests,
- **coverage** (float):  coverage percentage


### Data Splits

Dataset contains three subsets (full_context | medium_context | small_context) corresponding to the amount of information of dependencies input to the model.


## Usage
You can load RepoExec dataset using datasets library: ```pip install datasets```

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("Fsoft-AIC/RepoExec")

# specific subset (e.g. full_context) 
dataset = load_dataset("Fsoft-AIC/RepoExec", split="full_context")
```


## Additional Information
### Other Resources:
- Github: https://github.com/FSoft-AI4Code/RepoExec
- Webpage: https://fsoft-ai4code.github.io/repoexec
- Leaderboard: https://repoexec.github.io
- Paper: https://arxiv.org/html/2406.11927v1


### Licensing Information
MIT License

### Citation Information

```
@article{nam2024repoexec,
  title={RepoExec: Evaluate Code Generation with a Repository-Level Executable Benchmark},
  author={Hai, Nam Le and Manh, Dung Nguyen and Bui, Nghi DQ},
  journal={arXiv preprint arXiv:2406.11927v1},
  year={2024}
}
```

### Contributions
This dataset is developed by [FSOFT AI4Code team](https://github.com/FSoft-AI4Code).
