"""Tests for TAC tools."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tac.context.knowledge import KnowledgeClient
from tac.context.memory import MemoryClient
from tac.models.knowledge import KnowledgeBase, KnowledgeChunkResult
from tac.models.session import ConversationSession
from tac.tools.base import (
    TACTool,
    _extract_schema_from_function,
    _is_optional,
    _type_to_json_schema,
    create_tool,
    function_tool,
)
from tac.tools.knowledge import create_knowledge_tool
from tac.tools.memory import create_memory_tool


class TestTACTool:
    """Test TACTool class."""

    def test_tac_tool_creation(self):
        """Test TACTool can be created with required fields."""

        def dummy_func(x: str) -> str:
            return x

        tool = TACTool(
            name="test_tool",
            description="A test tool",
            params_json_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            _raw_implementation=dummy_func,
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        # implementation is now a property that returns a clean callable
        assert callable(tool.implementation)

    def test_to_openai_format(self):
        """Test conversion to OpenAI function format."""

        def dummy_func(x: str) -> str:
            return x

        tool = TACTool(
            name="test_tool",
            description="A test tool",
            params_json_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            _raw_implementation=dummy_func,
        )

        openai_format = tool.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "test_tool"
        assert openai_format["function"]["description"] == "A test tool"
        assert openai_format["function"]["parameters"]["type"] == "object"
        assert "x" in openai_format["function"]["parameters"]["properties"]

    def test_to_anthropic_format(self):
        """Test conversion to Anthropic tool format."""

        def dummy_func(x: str) -> str:
            return x

        tool = TACTool(
            name="test_tool",
            description="A test tool",
            params_json_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            _raw_implementation=dummy_func,
        )

        anthropic_format = tool.to_anthropic_format()

        assert anthropic_format["name"] == "test_tool"
        assert anthropic_format["description"] == "A test tool"
        assert anthropic_format["input_schema"]["type"] == "object"
        assert "x" in anthropic_format["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_to_openai_agents_sdk_tool(self):
        """Test conversion to OpenAI Agents SDK FunctionTool with live invocation."""
        from agents import FunctionTool

        async def dummy_func(x: str) -> dict[str, str]:
            return {"echo": x}

        tool = TACTool(
            name="test_tool",
            description="A test tool",
            params_json_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            _raw_implementation=dummy_func,
        )

        sdk_tool = tool.to_openai_agents_sdk_tool()

        assert isinstance(sdk_tool, FunctionTool)
        assert sdk_tool.name == "test_tool"
        assert sdk_tool.description == "A test tool"
        assert sdk_tool.params_json_schema == tool.params_json_schema

        result_json = await sdk_tool.on_invoke_tool(None, '{"x": "hello"}')
        assert json.loads(result_json) == {"echo": "hello"}

    def test_to_json(self):
        """Test conversion to JSON string."""

        def dummy_func(x: str) -> str:
            return x

        tool = TACTool(
            name="test_tool",
            description="A test tool",
            params_json_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            _raw_implementation=dummy_func,
        )

        json_str = tool.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "function"
        assert parsed["function"]["name"] == "test_tool"


class TestFunctionTool:
    """Test function_tool decorator."""

    def test_basic_function_tool(self):
        """Test basic function_tool decoration."""

        @function_tool()
        def simple_tool(message: str) -> str:
            """Send a simple message."""
            return f"Sent: {message}"

        assert isinstance(simple_tool, TACTool)
        assert simple_tool.name == "simple_tool"
        assert simple_tool.description == "Send a simple message."
        assert simple_tool.params_json_schema["type"] == "object"
        assert "message" in simple_tool.params_json_schema["properties"]
        assert simple_tool.params_json_schema["properties"]["message"]["type"] == "string"
        assert simple_tool.params_json_schema["required"] == ["message"]

    def test_function_tool_with_name_override(self):
        """Test function_tool with custom name."""

        @function_tool(name="custom_name")
        def simple_tool(message: str) -> str:
            """Send a simple message."""
            return f"Sent: {message}"

        assert simple_tool.name == "custom_name"

    def test_function_tool_with_description_override(self):
        """Test function_tool with custom description."""

        @function_tool(description="Custom description")
        def simple_tool(message: str) -> str:
            """Send a simple message."""
            return f"Sent: {message}"

        assert simple_tool.description == "Custom description"

    def test_function_tool_without_docstring_fails(self):
        """Test function_tool without docstring raises error."""
        with pytest.raises(ValueError, match="must have a docstring or description"):

            @function_tool()
            def no_docstring(message: str) -> str:
                return message

    def test_function_tool_with_optional_params(self):
        """Test function_tool with optional parameters."""

        @function_tool()
        def optional_tool(required: str, optional: str | None = None) -> str:
            """Tool with optional parameters."""
            return f"{required} - {optional}"

        assert "required" in optional_tool.params_json_schema["required"]
        assert "optional" not in optional_tool.params_json_schema["required"]

    def test_function_tool_with_default_values(self):
        """Test function_tool with default values."""

        @function_tool()
        def default_tool(message: str, priority: int = 1) -> str:
            """Tool with default values."""
            return f"{message} (priority: {priority})"

        assert "message" in default_tool.params_json_schema["required"]
        assert "priority" not in default_tool.params_json_schema["required"]

    @pytest.mark.asyncio
    async def test_function_tool_execution(self):
        """Test that decorated function can still be executed."""

        @function_tool()
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        # implementation property returns an async callable
        result = await add_numbers.implementation(5, 3)
        assert result == 8

    def test_function_tool_with_list_params(self):
        """Test function_tool with list parameters."""

        @function_tool()
        def list_tool(items: list[str]) -> str:
            """Process a list of items."""
            return ", ".join(items)

        schema = list_tool.params_json_schema
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"]["type"] == "string"

    def test_function_tool_with_multiple_types(self):
        """Test function_tool with various parameter types."""

        @function_tool()
        def multi_type_tool(
            text: str, number: int, decimal: float, flag: bool, items: list[str]
        ) -> dict:
            """Tool with multiple parameter types."""
            return {
                "text": text,
                "number": number,
                "decimal": decimal,
                "flag": flag,
                "items": items,
            }

        props = multi_type_tool.params_json_schema["properties"]
        assert props["text"]["type"] == "string"
        assert props["number"]["type"] == "integer"
        assert props["decimal"]["type"] == "number"
        assert props["flag"]["type"] == "boolean"
        assert props["items"]["type"] == "array"

    def test_function_tool_with_literal_and_enum(self):
        """Test function_tool with Literal types (enhanced capability)."""
        from enum import Enum
        from typing import Literal

        class Priority(str, Enum):
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"

        @function_tool()
        def send_notification(
            channel: Literal["sms", "voice", "email"],
            message: str,
            priority: Priority = Priority.LOW,
        ) -> dict:
            """Send a notification through specified channel."""
            return {"channel": channel, "message": message, "priority": priority.value}

        # Verify channel parameter has enum constraint
        channel_schema = send_notification.params_json_schema["properties"]["channel"]
        assert channel_schema["type"] == "string"
        assert set(channel_schema["enum"]) == {"sms", "voice", "email"}

        # Verify priority parameter has enum constraint
        priority_schema = send_notification.params_json_schema["properties"]["priority"]
        assert priority_schema["type"] == "string"
        assert set(priority_schema["enum"]) == {"low", "medium", "high"}

        # Verify required fields
        assert "channel" in send_notification.params_json_schema["required"]
        assert "message" in send_notification.params_json_schema["required"]
        assert "priority" not in send_notification.params_json_schema["required"]


class TestCreateTool:
    """Test create_tool function."""

    @pytest.mark.asyncio
    async def test_create_tool_manually(self):
        """Test manual tool creation with explicit schema."""

        def custom_impl(x: str) -> str:
            return x.upper()

        tool = create_tool(
            name="manual_tool",
            description="Manually created tool",
            params_json_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            implementation=custom_impl,
        )

        assert isinstance(tool, TACTool)
        assert tool.name == "manual_tool"
        assert tool.description == "Manually created tool"
        # implementation property returns an async callable
        result = await tool.implementation(x="test")
        assert result == "TEST"


class TestTypeToJsonSchema:
    """Test _type_to_json_schema function."""

    def test_basic_types(self):
        """Test conversion of basic Python types."""
        assert _type_to_json_schema(str) == {"type": "string"}
        assert _type_to_json_schema(int) == {"type": "integer"}
        assert _type_to_json_schema(float) == {"type": "number"}
        assert _type_to_json_schema(bool) == {"type": "boolean"}

    def test_optional_type(self):
        """Test conversion of Optional types - unwraps to base type."""
        schema = _type_to_json_schema(str | None)
        # Optional types are unwrapped to base type (optionality tracked via 'required' array)
        assert schema == {"type": "string"}

    def test_list_type(self):
        """Test conversion of List types."""
        schema = _type_to_json_schema(list[str])
        assert schema == {"type": "array", "items": {"type": "string"}}

    def test_dict_type(self):
        """Test conversion of dict type."""
        schema = _type_to_json_schema(dict)
        # TypeAdapter generates more complete schema with additionalProperties
        assert schema == {"type": "object", "additionalProperties": True}

    def test_plain_list_type(self):
        """Test conversion of plain list without type parameter."""
        schema = _type_to_json_schema(list)
        # TypeAdapter generates more complete schema with items field
        assert schema == {"type": "array", "items": {}}

    def test_literal_type(self):
        """Test conversion of Literal types (new capability with TypeAdapter)."""
        from typing import Literal

        schema = _type_to_json_schema(Literal["sms", "voice", "email"])
        assert schema["type"] == "string"
        assert set(schema["enum"]) == {"sms", "voice", "email"}

    def test_literal_int_type(self):
        """Test conversion of Literal with integers."""
        from typing import Literal

        schema = _type_to_json_schema(Literal[1, 2, 3])
        assert "enum" in schema
        assert set(schema["enum"]) == {1, 2, 3}

    def test_complex_union_type(self):
        """Test conversion of complex Union types (new capability)."""
        schema = _type_to_json_schema(str | int)
        # TypeAdapter handles complex unions with anyOf
        assert "anyOf" in schema
        types = [item.get("type") for item in schema["anyOf"]]
        assert "string" in types
        assert "integer" in types

    def test_enum_type(self):
        """Test conversion of Enum types (new capability with TypeAdapter)."""
        from enum import Enum

        class Channel(str, Enum):
            SMS = "sms"
            VOICE = "voice"
            EMAIL = "email"

        schema = _type_to_json_schema(Channel)
        assert schema["type"] == "string"
        assert set(schema["enum"]) == {"sms", "voice", "email"}


class TestIsOptional:
    """Test _is_optional function."""

    def test_optional_type_returns_true(self):
        """Test that Optional[T] is detected as optional."""
        assert _is_optional(str | None) is True

    def test_non_optional_type_returns_false(self):
        """Test that non-Optional types are not detected as optional."""
        assert _is_optional(str) is False
        assert _is_optional(int) is False


class TestExtractSchemaFromFunction:
    """Test _extract_schema_from_function."""

    def test_extract_simple_function_schema(self):
        """Test schema extraction from simple function."""

        def simple_func(name: str, age: int) -> str:
            return f"{name} is {age}"

        schema = _extract_schema_from_function(simple_func)

        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["age"]["type"] == "integer"
        assert set(schema["required"]) == {"name", "age"}

    def test_extract_schema_with_optional(self):
        """Test schema extraction with optional parameters."""

        def optional_func(required: str, optional: str | None = None) -> str:
            return required

        schema = _extract_schema_from_function(optional_func)

        assert "required" in schema["required"]
        assert "optional" not in schema["required"]

    def test_extract_schema_with_defaults(self):
        """Test schema extraction with default values."""

        def default_func(name: str, count: int = 10) -> str:
            return name

        schema = _extract_schema_from_function(default_func)

        assert "name" in schema["required"]
        assert "count" not in schema["required"]

    def test_extract_schema_ignores_self(self):
        """Test that schema extraction ignores 'self' parameter."""

        class TestClass:
            def method(self, value: str) -> str:
                return value

        schema = _extract_schema_from_function(TestClass.method)

        assert "self" not in schema["properties"]
        assert "value" in schema["properties"]


class TestMemoryTools:
    """Test create_memory_tool function."""

    def test_create_memory_tool_returns_tool(self):
        """Test that create_memory_tool returns a TACTool."""
        # Create mock MemoryClient
        mock_memory_client = MagicMock(spec=MemoryClient)

        session = ConversationSession(
            profile_id="prof_123", conversation_id="conv_123", channel="SMS"
        )

        tool = create_memory_tool(mock_memory_client, session)

        assert isinstance(tool, TACTool)

    def test_memory_tool_has_correct_schema(self):
        """Test that memory tool has correct schema."""
        # Create mock MemoryClient
        mock_memory_client = MagicMock(spec=MemoryClient)

        session = ConversationSession(
            profile_id="prof_123", conversation_id="conv_123", channel="SMS"
        )

        memory_tool = create_memory_tool(mock_memory_client, session)

        assert memory_tool.name == "retrieve_profile_memory"
        assert "query" in memory_tool.params_json_schema["properties"]
        assert memory_tool.params_json_schema["properties"]["query"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_memory_tool_makes_api_call(self):
        """Test that memory tool makes correct API call via MemoryClient."""
        # Create mock MemoryClient with async method
        mock_memory_client = MagicMock(spec=MemoryClient)

        async def mock_retrieve(*args, **kwargs):
            from tac.models.memory import MemoryRetrievalResponse

            return MemoryRetrievalResponse()

        mock_memory_client.retrieve_memory = MagicMock(side_effect=mock_retrieve)

        session = ConversationSession(
            profile_id="prof_123", conversation_id="conv_123", channel="SMS"
        )

        memory_tool = create_memory_tool(mock_memory_client, session)

        result = await memory_tool(query="test query")

        # Verify MemoryClient call
        mock_memory_client.retrieve_memory.assert_called_once_with(
            profile_id="prof_123",
            query="test query",
        )

        # Verify result structure
        assert "observations" in result
        assert "summaries" in result
        assert "communications" in result
        assert "meta" in result

    def test_memory_tool_uses_injected_config(self):
        """Test that memory tools use injected dependencies."""
        # Create mock MemoryClients
        mock_client1 = MagicMock(spec=MemoryClient)
        mock_client2 = MagicMock(spec=MemoryClient)

        session1 = ConversationSession(profile_id="prof_1", conversation_id="conv_1", channel="SMS")
        session2 = ConversationSession(profile_id="prof_2", conversation_id="conv_2", channel="SMS")

        tool1 = create_memory_tool(mock_client1, session1)
        tool2 = create_memory_tool(mock_client2, session2)

        # Tools share the same tool name and raw implementation but have different injected args
        assert tool1.name == tool2.name  # Same tool name
        assert tool1._raw_implementation == tool2._raw_implementation  # Same raw function
        client1 = tool1._injected_args["conversation_memory_client"]
        client2 = tool2._injected_args["conversation_memory_client"]
        assert client1 != client2
        assert tool1._injected_args["profile_id"] == "prof_1"
        assert tool2._injected_args["profile_id"] == "prof_2"

    @pytest.mark.asyncio
    async def test_memory_tool_injected_params_not_in_schema(self):
        """Test that injected parameters are not exposed in tool schema."""
        # Create mock MemoryClient with async method
        mock_memory_client = MagicMock(spec=MemoryClient)

        async def mock_retrieve(*args, **kwargs):
            from tac.models.memory import MemoryRetrievalResponse

            return MemoryRetrievalResponse()

        mock_memory_client.retrieve_memory = MagicMock(side_effect=mock_retrieve)

        session = ConversationSession(
            profile_id="prof_123", conversation_id="conv_123", channel="SMS"
        )

        memory_tool = create_memory_tool(mock_memory_client, session)

        # Only query should be in schema, not the injected params
        assert "query" in memory_tool.params_json_schema["properties"]
        assert "conversation_memory_client" not in memory_tool.params_json_schema["properties"]
        assert "profile_id" not in memory_tool.params_json_schema["properties"]

        # Verify tool still works
        await memory_tool(query="test query")
        mock_memory_client.retrieve_memory.assert_called_once()

    def test_memory_tool_type_validation(self):
        """Test that configure_injection validates types correctly."""
        from typing import Annotated

        from tac.tools.base import InjectedToolArg, function_tool

        # Create a tool with typed injected parameters
        async def test_tool(
            query: str,
            client: Annotated[MemoryClient, InjectedToolArg],
            count: Annotated[int, InjectedToolArg],
        ) -> str:
            """Test tool with typed injections."""
            return "result"

        tool = function_tool()(test_tool)

        # Valid types should work
        mock_client = MagicMock(spec=MemoryClient)
        tool.configure_injection(client=mock_client, count=5)

        # Wrong type should raise TypeError
        with pytest.raises(TypeError, match="Type mismatch.*count"):
            tool.configure_injection(count="wrong")

        # Unknown parameter should raise ValueError
        with pytest.raises(ValueError, match="Unknown injected parameter 'unknown'"):
            tool.configure_injection(unknown="value")

    def test_type_validation_with_generic_types(self):
        """Test that configure_injection validates generic types correctly."""
        from typing import Annotated, Any

        from tac.tools.base import InjectedToolArg, function_tool

        # Create a tool with generic type annotations
        async def test_tool(
            query: str,
            items: Annotated[list[str], InjectedToolArg],
            config: Annotated[dict[str, Any], InjectedToolArg],
        ) -> str:
            """Test tool with generic type injections."""
            return "result"

        tool = function_tool()(test_tool)

        # Valid generic types should work
        tool.configure_injection(items=["a", "b", "c"], config={"key": "value"})

        # Wrong type for list[str] should raise TypeError
        with pytest.raises(TypeError, match="Type mismatch.*items"):
            tool.configure_injection(items={"wrong": "type"})

        # Wrong item type in list should raise TypeError
        with pytest.raises(TypeError, match="Type mismatch.*items"):
            tool.configure_injection(items=[1, 2, 3])  # list[int] not list[str]

        # Wrong type for dict should raise TypeError
        with pytest.raises(TypeError, match="Type mismatch.*config"):
            tool.configure_injection(config=["not", "a", "dict"])

    def test_type_validation_with_pydantic_models(self):
        """Test that configure_injection validates Pydantic models correctly."""
        from typing import Annotated

        from pydantic import BaseModel

        from tac.tools.base import InjectedToolArg, function_tool

        class TestConfig(BaseModel):
            name: str
            value: int

        # Create a tool with Pydantic model annotation
        async def test_tool(
            query: str,
            config: Annotated[TestConfig, InjectedToolArg],
        ) -> str:
            """Test tool with Pydantic model injection."""
            return "result"

        tool = function_tool()(test_tool)

        # Valid Pydantic model should work
        valid_config = TestConfig(name="test", value=42)
        tool.configure_injection(config=valid_config)

        # Pydantic accepts dicts and coerces them to models (this is expected behavior)
        tool.configure_injection(config={"name": "test", "value": 42})

        # Invalid dict structure should raise TypeError
        with pytest.raises(TypeError, match="Type mismatch.*config"):
            tool.configure_injection(config={"wrong": "fields"})

        # Completely wrong type should raise TypeError
        with pytest.raises(TypeError, match="Type mismatch.*config"):
            tool.configure_injection(config="not a model")

        # Invalid field type should raise TypeError
        with pytest.raises(TypeError, match="Type mismatch.*config"):
            tool.configure_injection(config={"name": "test", "value": "not_an_int"})

    def test_type_validation_with_optional_generic_types(self):
        """Test that configure_injection validates Optional generic types correctly."""
        from typing import Annotated

        from tac.tools.base import InjectedToolArg, function_tool

        # Create a tool with Optional generic type annotations
        async def test_tool(
            query: str,
            items: Annotated[list[str] | None, InjectedToolArg],
        ) -> str:
            """Test tool with optional generic type injections."""
            return "result"

        tool = function_tool()(test_tool)

        # Valid list[str] should work
        tool.configure_injection(items=["a", "b", "c"])

        # None should work for Optional
        tool.configure_injection(items=None)

        # Wrong type should raise TypeError
        with pytest.raises(TypeError, match="Type mismatch.*items"):
            tool.configure_injection(items={"wrong": "type"})


class TestKnowledgeTools:
    """Test create_knowledge_tool function."""

    @pytest.mark.asyncio
    async def test_create_knowledge_tool_returns_tac_tool(self):
        """Test that create_knowledge_tool returns a TACTool."""
        # Create mock KnowledgeClient
        mock_knowledge_client = MagicMock(spec=KnowledgeClient)

        knowledge_base = KnowledgeBase(
            id="know_knowledgebase_00000000000000000000000000",
            display_name="product-faq",
            description="Frequently asked questions about products",
            status="ACTIVE",
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T11:45:00Z",
            version=1,
        )

        tool = await create_knowledge_tool(
            mock_knowledge_client,
            knowledge_base.id,
            name="search_product_faq",
            description=knowledge_base.description,
        )

        assert isinstance(tool, TACTool)

    @pytest.mark.asyncio
    async def test_knowledge_tool_default_name_and_description(self):
        """Test that knowledge tool uses default name and description from KB metadata."""
        # Create mock KnowledgeClient
        mock_knowledge_client = MagicMock(spec=KnowledgeClient)

        knowledge_base = KnowledgeBase(
            id="know_knowledgebase_00000000000000000000000000",
            display_name="product-faq",
            description="Frequently asked questions about products",
            status="ACTIVE",
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T11:45:00Z",
            version=1,
        )

        # Mock get_knowledge_base to return the knowledge base
        mock_knowledge_client.get_knowledge_base = AsyncMock(return_value=knowledge_base)

        # Call without name/description to trigger the KB-metadata defaults
        tool = await create_knowledge_tool(
            mock_knowledge_client,
            knowledge_base.id,
        )

        # Verify get_knowledge_base was called
        mock_knowledge_client.get_knowledge_base.assert_called_once_with(knowledge_base.id)

        # Verify default name is generated from display_name
        assert tool.name == "search_product_faq"
        # Verify default description includes the KB description
        assert "Frequently asked questions about products" in tool.description
        assert "The input MUST be a question in the form of a string." in tool.description

    @pytest.mark.asyncio
    async def test_knowledge_tool_custom_name_and_description(self):
        """Test that knowledge tool respects custom name and description."""
        # Create mock KnowledgeClient
        mock_knowledge_client = MagicMock(spec=KnowledgeClient)

        knowledge_base_id = "know_knowledgebase_00000000000000000000000000"

        tool = await create_knowledge_tool(
            mock_knowledge_client,
            knowledge_base_id,
            name="custom_product_search",
            description="Search product documentation",
        )

        assert tool.name == "custom_product_search"
        assert tool.description == "Search product documentation"

    @pytest.mark.asyncio
    async def test_knowledge_tool_custom_top_k(self):
        """Test that knowledge tool respects custom top-K value."""
        # Create mock KnowledgeClient
        mock_knowledge_client = MagicMock(spec=KnowledgeClient)

        knowledge_base_id = "know_knowledgebase_00000000000000000000000000"

        tool = await create_knowledge_tool(
            mock_knowledge_client,
            knowledge_base_id,
            name="test_tool",
            description="Test description",
            top_k=10,
        )

        # We can't directly access top_k from outside,
        # but we can verify it's used in the API call via mocking
        assert isinstance(tool, TACTool)

    @pytest.mark.asyncio
    async def test_knowledge_tool_has_correct_schema(self):
        """Test that knowledge tool has correct parameter schema."""
        # Create mock KnowledgeClient
        mock_knowledge_client = MagicMock(spec=KnowledgeClient)

        knowledge_base = KnowledgeBase(
            id="know_knowledgebase_00000000000000000000000000",
            display_name="product-faq",
            description="Frequently asked questions about products",
            status="ACTIVE",
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T11:45:00Z",
            version=1,
        )

        tool = await create_knowledge_tool(
            mock_knowledge_client,
            knowledge_base.id,
            name="search_product_faq",
            description=knowledge_base.description,
        )

        assert "query" in tool.params_json_schema["properties"]
        assert tool.params_json_schema["properties"]["query"]["type"] == "string"
        assert "query" in tool.params_json_schema["required"]

    @pytest.mark.asyncio
    async def test_knowledge_tool_makes_api_call(self):
        """Test that knowledge tool makes correct API call via KnowledgeClient."""
        # Create mock KnowledgeClient with async method
        mock_knowledge_client = MagicMock(spec=KnowledgeClient)

        async def mock_search(*args, **kwargs):
            return [
                KnowledgeChunkResult(
                    content="Answer 1",
                    knowledge_id="know_knowledge_00000000000000000000000001",
                    created_at="2024-01-15T10:30:00Z",
                    score=0.95,
                ),
                KnowledgeChunkResult(
                    content="Answer 2",
                    knowledge_id="know_knowledge_00000000000000000000000002",
                    created_at="2024-01-15T10:30:00Z",
                    score=0.87,
                ),
            ]

        mock_knowledge_client.search_knowledge_base = MagicMock(side_effect=mock_search)

        knowledge_base = KnowledgeBase(
            id="know_knowledgebase_00000000000000000000000000",
            display_name="product-faq",
            description="Frequently asked questions about products",
            status="ACTIVE",
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T11:45:00Z",
            version=1,
        )

        tool = await create_knowledge_tool(
            mock_knowledge_client,
            knowledge_base.id,
            name="search_product_faq",
            description=knowledge_base.description,
        )
        result = await tool(query="What is the return policy?")

        # Verify KnowledgeClient call
        mock_knowledge_client.search_knowledge_base.assert_called_once_with(
            knowledge_base_id="know_knowledgebase_00000000000000000000000000",
            query="What is the return policy?",
            top_k=5,  # Default value
        )

        # Verify result
        assert len(result) == 2
        assert result[0].content == "Answer 1"
        assert result[1].content == "Answer 2"

    @pytest.mark.asyncio
    async def test_knowledge_tool_uses_custom_top_k(self):
        """Test that knowledge tool uses custom top-K value in API call."""
        # Create mock KnowledgeClient with async method
        mock_knowledge_client = MagicMock(spec=KnowledgeClient)

        async def mock_search(*args, **kwargs):
            return []

        mock_knowledge_client.search_knowledge_base = MagicMock(side_effect=mock_search)

        knowledge_base_id = "know_knowledgebase_00000000000000000000000000"

        tool = await create_knowledge_tool(
            mock_knowledge_client,
            knowledge_base_id,
            name="test_tool",
            description="Test description",
            top_k=10,
        )
        await tool(query="test query")

        # Verify top-K value
        mock_knowledge_client.search_knowledge_base.assert_called_once_with(
            knowledge_base_id="know_knowledgebase_00000000000000000000000000",
            query="test query",
            top_k=10,
        )

    @pytest.mark.asyncio
    async def test_knowledge_tool_uses_injected_config(self):
        """Test that knowledge tools use injected dependencies."""
        # Create mock KnowledgeClients
        mock_client1 = MagicMock(spec=KnowledgeClient)
        mock_client2 = MagicMock(spec=KnowledgeClient)

        knowledge_base1 = KnowledgeBase(
            id="know_knowledgebase_00000000000000000000000001",
            display_name="faq-1",
            description="First FAQ",
            status="ACTIVE",
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T11:45:00Z",
            version=1,
        )
        knowledge_base2 = KnowledgeBase(
            id="know_knowledgebase_00000000000000000000000002",
            display_name="faq-2",
            description="Second FAQ",
            status="ACTIVE",
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T11:45:00Z",
            version=1,
        )

        tool1 = await create_knowledge_tool(
            mock_client1,
            knowledge_base1.id,
            name="tool1",
            description="Tool 1",
        )
        tool2 = await create_knowledge_tool(
            mock_client2,
            knowledge_base2.id,
            name="tool2",
            description="Tool 2",
        )

        # Tools share the same raw implementation function but have different injected args
        assert tool1._raw_implementation == tool2._raw_implementation  # Same raw function
        assert tool1._injected_args["knowledge_client"] != tool2._injected_args["knowledge_client"]
        assert (
            tool1._injected_args["knowledge_base_id"]
            == "know_knowledgebase_00000000000000000000000001"
        )
        assert (
            tool2._injected_args["knowledge_base_id"]
            == "know_knowledgebase_00000000000000000000000002"
        )

    @pytest.mark.asyncio
    async def test_knowledge_tool_injected_params_not_in_schema(self):
        """Test that injected parameters are not exposed in tool schema."""
        # Create mock KnowledgeClient with async method
        mock_knowledge_client = MagicMock(spec=KnowledgeClient)

        async def mock_search(*args, **kwargs):
            return []

        mock_knowledge_client.search_knowledge_base = MagicMock(side_effect=mock_search)

        knowledge_base = KnowledgeBase(
            id="know_knowledgebase_00000000000000000000000000",
            display_name="product-faq",
            description="Frequently asked questions about products",
            status="ACTIVE",
            created_at="2024-01-15T10:30:00Z",
            updated_at="2024-01-15T11:45:00Z",
            version=1,
        )

        tool = await create_knowledge_tool(
            mock_knowledge_client,
            knowledge_base.id,
            name="search_product_faq",
            description=knowledge_base.description,
        )

        # Only query should be in schema, not the injected params
        assert "query" in tool.params_json_schema["properties"]
        assert "knowledge_client" not in tool.params_json_schema["properties"]
        assert "knowledge_base_id" not in tool.params_json_schema["properties"]
        assert "top_k" not in tool.params_json_schema["properties"]

        # Verify tool still works
        await tool(query="test query")
        mock_knowledge_client.search_knowledge_base.assert_called_once()
