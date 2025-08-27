from src.parser.base_parser import ElementType
from src.parser.plugin_registry import LanguagePluginRegistry


def test_heritage_and_methods_extraction() -> None:
    plugin = LanguagePluginRegistry.get_plugin("typescript")
    if not plugin:
        return
    parser = plugin.create_parser()

    content = (
        "abstract class BaseService { constructor(){} abstract getData(): any; }\n"
        "class UserService extends BaseService implements EventEmitter { constructor(){} fetchUser(){} createUser(){} }\n"
    )

    res = parser.parse_string(content, filename="test1.ts")

    classes = {c.name: c for c in res.find_elements(ElementType.CLASS)}

    assert "BaseService" in classes
    assert "UserService" in classes

    us_meta = classes["UserService"].metadata
    assert "BaseService" in us_meta.get("base_classes", [])
    assert "EventEmitter" in us_meta.get("interfaces", [])

    methods = [
        m.name
        for m in res.find_elements(ElementType.METHOD)
        if m.parent and m.parent.name == "UserService"
    ]
    assert set(methods) >= {"constructor", "fetchUser", "createUser"}


def test_arrow_function_generic_detection() -> None:
    plugin = LanguagePluginRegistry.get_plugin("typescript")
    if not plugin:
        return
    parser = plugin.create_parser()

    content = (
        "const filterUsersByRole = <T extends User>(users: T[], roleName: string): T[] => { return users.filter(u => u); };\n"
        "const simple = (x) => x;\n"
    )

    res = parser.parse_string(content, filename="test2.ts")
    functions = [f.name for f in res.find_elements(ElementType.FUNCTION)]

    assert "filterUsersByRole" in functions
    assert "simple" in functions
