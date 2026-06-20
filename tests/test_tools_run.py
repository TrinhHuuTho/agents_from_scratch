import asyncio

import agent_from_scratch.tools as tools_module
from agent_from_scratch.tools import create_default_tool_registry


def test_tools_run_basic():
    registry = create_default_tool_registry()

    weather_tool = registry["get_weather"]
    translate_tool = registry["translate"]

    weather_res = asyncio.run(weather_tool.ainvoke({"city": "Hanoi"}))
    assert isinstance(weather_res, str)
    assert weather_res

    translate_res = asyncio.run(
        translate_tool.ainvoke({"text": "Bạn có khỏe không", "target_language": "en"})
    )
    assert isinstance(translate_res, str)
    assert translate_res


def test_translate_supports_async_translator(monkeypatch):
    class FakeTranslation:
        text = "How are you?"

    class FakeTranslator:
        async def translate(self, text, dest):
            assert text == "Bạn có khỏe không"
            assert dest == "en"
            return FakeTranslation()

    monkeypatch.setattr(tools_module, "Translator", FakeTranslator)

    result = asyncio.run(
        tools_module.translate.ainvoke(
            {"text": "Bạn có khỏe không", "target_language": "en"}
        )
    )

    assert result == "'Bạn có khỏe không' in en is 'How are you?'"


def test_get_weather_runs_with_fake_client(monkeypatch):
    class FakeForecast:
        temperature = 32
        description = "Partly cloudy"
        kind = "Partly cloudy"

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, city):
            assert city == "Hanoi"
            return FakeForecast()

    monkeypatch.setattr(tools_module.python_weather, "Client", FakeClient)

    result = asyncio.run(tools_module.get_weather.ainvoke({"city": "Hanoi"}))

    assert result == '{"temperature": 32, "sky_text": "Partly cloudy"}'
