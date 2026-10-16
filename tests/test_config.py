"""Tests for loading the CLI configuration."""

from __future__ import annotations

import logging
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

from mcp_to_cli_kit.config import (
    CliConfig,
    ConfigError,
    GroupMount,
    PrefixMount,
    ServerConfig,
    ToolGroup,
    default_config_path,
    env_prefix_for,
    load_config,
    resolve_config_path,
    save_config,
)

BIN = "example-tool"

DEFAULT: dict[str, Any] = {
    "help": "Example tool",
    "servers": {
        "alpha": {
            "url": "https://alpha.example.com/mcp",
            "headers": {"X-Team": "core", "X-Client": "default"},
            "timeout": 10,
        },
        "beta-two": {
            "url": "https://beta.example.com/mcp",
            "mount": {"group": "beta"},
            "tool_groups": [{"prefix": "search_", "group": "search", "help": "Search tools"}],
            "hidden_tools": ["internal_debug"],
        },
    },
}


@pytest.fixture(autouse=True)
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the home directory at an empty temporary one."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def write_yaml(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestPaths:
    def test_default_path_is_under_dot_config(self, home: Path) -> None:
        assert default_config_path(BIN) == home / ".config" / BIN / "config.yaml"

    def test_explicit_path_wins_and_expands_the_home(self, home: Path) -> None:
        assert resolve_config_path(BIN, "~/elsewhere.yaml") == home / "elsewhere.yaml"

    def test_no_explicit_path_falls_back_to_the_default(self) -> None:
        assert resolve_config_path(BIN, None) == default_config_path(BIN)
        assert resolve_config_path(BIN, "") == default_config_path(BIN)

    def test_env_prefix_is_derived_from_the_bin_name(self) -> None:
        assert env_prefix_for("example-tool") == "EXAMPLE_TOOL_"
        assert env_prefix_for("ex.tool2") == "EX_TOOL2_"


class TestSources:
    def test_default_alone(self) -> None:
        loaded = load_config(BIN, default=DEFAULT, environ={})

        assert loaded.file_loaded is False
        assert loaded.config.help == "Example tool"
        assert loaded.config.servers["alpha"].timeout == 10
        assert loaded.config.servers["alpha"].mount == "root"
        assert loaded.config.servers["beta-two"].mount == GroupMount(group="beta")

    def test_file_alone(self, home: Path) -> None:
        write_yaml(default_config_path(BIN), {"servers": {"gamma": {"url": "http://localhost:8000/mcp"}}})

        loaded = load_config(BIN, environ={})

        assert loaded.file_loaded is True
        assert loaded.path == default_config_path(BIN)
        assert list(loaded.config.servers) == ["gamma"]

    def test_no_default_and_no_file_names_the_expected_path(self, home: Path) -> None:
        with pytest.raises(ConfigError) as error:
            load_config(BIN, environ={})

        assert str(default_config_path(BIN)) in str(error.value)
        assert "servers:" in str(error.value)

    def test_env_alone_is_not_enough(self) -> None:
        with pytest.raises(ConfigError, match="has no configuration"):
            load_config(BIN, environ={"EXAMPLE_TOOL_SERVERS__ALPHA__URL": "https://alpha.example.com/mcp"})

    def test_explicit_path_must_exist_even_with_a_default(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.yaml"

        with pytest.raises(ConfigError, match=str(missing)):
            load_config(BIN, default=DEFAULT, config_path=missing, environ={})

    def test_explicit_path_to_a_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="is a directory, not a file"):
            load_config(BIN, default=DEFAULT, config_path=tmp_path, environ={})

    def test_default_path_that_is_a_directory(self) -> None:
        default_config_path(BIN).mkdir(parents=True)

        with pytest.raises(ConfigError, match="is a directory, not a file"):
            load_config(BIN, default=DEFAULT, environ={})

    def test_a_file_cannot_remove_a_server_of_the_default(self) -> None:
        """Documented limitation: layers merge, so a server in the default always stays."""
        write_yaml(default_config_path(BIN), {"servers": {"gamma": {"url": "https://gamma.example.com/mcp"}}})

        config = load_config(BIN, default=DEFAULT, environ={}).config

        assert set(config.servers) == {"alpha", "beta-two", "gamma"}

    def test_explicit_path_replaces_the_default_path(self, tmp_path: Path) -> None:
        write_yaml(default_config_path(BIN), {"servers": {"alpha": {"timeout": 99}}})
        explicit = write_yaml(tmp_path / "other.yaml", {"servers": {"alpha": {"timeout": 42}}})

        loaded = load_config(BIN, default=DEFAULT, config_path=explicit, environ={})

        assert loaded.path == explicit
        assert loaded.config.servers["alpha"].timeout == 42

    def test_empty_file_is_an_empty_layer(self) -> None:
        default_config_path(BIN).parent.mkdir(parents=True)
        default_config_path(BIN).write_text("", encoding="utf-8")

        loaded = load_config(BIN, default=DEFAULT, environ={})

        assert loaded.file_loaded is True
        assert loaded.config.servers["alpha"].timeout == 10

    def test_malformed_yaml_names_the_file(self) -> None:
        default_config_path(BIN).parent.mkdir(parents=True)
        default_config_path(BIN).write_text("servers: [unclosed", encoding="utf-8")

        with pytest.raises(ConfigError, match="not valid YAML"):
            load_config(BIN, default=DEFAULT, environ={})

    def test_top_level_must_be_a_mapping(self) -> None:
        write_yaml(default_config_path(BIN), ["a", "b"])

        with pytest.raises(ConfigError, match="mapping at the top level"):
            load_config(BIN, default=DEFAULT, environ={})

    def test_default_may_be_a_model(self) -> None:
        default = CliConfig(servers={"alpha": ServerConfig(url="https://alpha.example.com/mcp")})
        write_yaml(default_config_path(BIN), {"servers": {"alpha": {"timeout": 5}}})

        loaded = load_config(BIN, default=default, environ={})

        assert loaded.config.servers["alpha"].url == "https://alpha.example.com/mcp"
        assert loaded.config.servers["alpha"].timeout == 5

    def test_the_default_is_not_mutated(self) -> None:
        write_yaml(default_config_path(BIN), {"servers": {"alpha": {"headers": {"X-Team": "file"}}}})

        load_config(BIN, default=DEFAULT, environ={"EXAMPLE_TOOL_SERVERS__ALPHA__TIMEOUT": "1"})

        assert DEFAULT["servers"]["alpha"]["headers"]["X-Team"] == "core"
        assert DEFAULT["servers"]["alpha"]["timeout"] == 10


class TestPrecedence:
    def test_file_overrides_one_key_of_one_server(self) -> None:
        write_yaml(default_config_path(BIN), {"servers": {"alpha": {"timeout": 20}}})

        config = load_config(BIN, default=DEFAULT, environ={}).config

        alpha = config.servers["alpha"]
        assert alpha.timeout == 20
        assert alpha.url == "https://alpha.example.com/mcp"
        assert alpha.headers == {"X-Team": "core", "X-Client": "default"}
        assert config.servers["beta-two"].hidden_tools == ["internal_debug"]

    def test_headers_merge_key_by_key(self) -> None:
        write_yaml(default_config_path(BIN), {"servers": {"alpha": {"headers": {"X-Team": "file", "X-New": "n"}}}})

        alpha = load_config(BIN, default=DEFAULT, environ={}).config.servers["alpha"]

        assert alpha.headers == {"X-Team": "file", "X-Client": "default", "X-New": "n"}

    def test_lists_are_replaced_not_merged(self) -> None:
        write_yaml(default_config_path(BIN), {"servers": {"beta-two": {"hidden_tools": ["other"]}}})

        beta = load_config(BIN, default=DEFAULT, environ={}).config.servers["beta-two"]

        assert beta.hidden_tools == ["other"]
        assert beta.tool_groups == [ToolGroup(prefix="search_", group="search", help="Search tools")]

    def test_mount_is_replaced_not_merged(self) -> None:
        write_yaml(default_config_path(BIN), {"servers": {"beta-two": {"mount": {"prefix": "b-"}}}})

        beta = load_config(BIN, default=DEFAULT, environ={}).config.servers["beta-two"]

        assert beta.mount == PrefixMount(prefix="b-")

    def test_file_adds_a_server(self) -> None:
        write_yaml(default_config_path(BIN), {"servers": {"gamma": {"url": "https://gamma.example.com/mcp"}}})

        config = load_config(BIN, default=DEFAULT, environ={}).config

        assert list(config.servers) == ["alpha", "beta-two", "gamma"]

    def test_env_overrides_the_file(self) -> None:
        write_yaml(default_config_path(BIN), {"servers": {"alpha": {"url": "https://file.example.com/mcp"}}})

        alpha = load_config(
            BIN, default=DEFAULT, environ={"EXAMPLE_TOOL_SERVERS__ALPHA__URL": "https://env.example.com/mcp"}
        ).config.servers["alpha"]

        assert alpha.url == "https://env.example.com/mcp"
        assert alpha.timeout == 10

    def test_env_overrides_one_header_of_one_server(self) -> None:
        write_yaml(default_config_path(BIN), {"servers": {"alpha": {"headers": {"X-Team": "file"}}}})

        config = load_config(
            BIN, default=DEFAULT, environ={"EXAMPLE_TOOL_SERVERS__ALPHA__HEADERS__X_TEAM": "env"}
        ).config

        assert config.servers["alpha"].headers == {"X-Team": "env", "X-Client": "default"}
        assert config.servers["beta-two"].headers == {}


class TestEnvironment:
    def load(self, **environ: str) -> CliConfig:
        return load_config(BIN, default=DEFAULT, environ=environ).config

    def test_server_names_match_with_dashes_and_any_case(self) -> None:
        config = self.load(EXAMPLE_TOOL_SERVERS__BETA_TWO__TIMEOUT="7")

        assert config.servers["beta-two"].timeout == 7
        assert list(config.servers) == ["alpha", "beta-two"]

    def test_new_header_name_gets_dashes(self) -> None:
        config = self.load(EXAMPLE_TOOL_SERVERS__ALPHA__HEADERS__X_API_KEY="secret")

        assert config.servers["alpha"].headers["X-Api-Key"] == "secret"

    def test_top_level_help(self) -> None:
        assert self.load(EXAMPLE_TOOL_HELP="From env").help == "From env"

    def test_lists_are_json(self) -> None:
        config = self.load(EXAMPLE_TOOL_SERVERS__BETA_TWO__HIDDEN_TOOLS='["a", "b"]')

        assert config.servers["beta-two"].hidden_tools == ["a", "b"]

    def test_invalid_json_is_an_error(self) -> None:
        with pytest.raises(ConfigError, match="EXAMPLE_TOOL_SERVERS__BETA_TWO__HIDDEN_TOOLS is not valid JSON"):
            self.load(EXAMPLE_TOOL_SERVERS__BETA_TWO__HIDDEN_TOOLS="[a, b")

    def test_a_comma_list_is_not_split(self) -> None:
        with pytest.raises(ConfigError, match="hidden_tools"):
            self.load(EXAMPLE_TOOL_SERVERS__BETA_TWO__HIDDEN_TOOLS="a,b")

    def test_mount_as_a_word_json_or_nested_key(self) -> None:
        assert self.load(EXAMPLE_TOOL_SERVERS__BETA_TWO__MOUNT="root").servers["beta-two"].mount == "root"
        assert self.load(EXAMPLE_TOOL_SERVERS__ALPHA__MOUNT='{"prefix": "a-"}').servers["alpha"].mount == PrefixMount(
            prefix="a-"
        )
        assert self.load(EXAMPLE_TOOL_SERVERS__BETA_TWO__MOUNT__PREFIX="b-").servers["beta-two"].mount == PrefixMount(
            prefix="b-"
        )

    def test_unknown_server_is_ignored_and_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG, logger="mcp_to_cli_kit.config")

        config = self.load(EXAMPLE_TOOL_SERVERS__GAMMA__URL="https://gamma.example.com/mcp")

        assert list(config.servers) == ["alpha", "beta-two"]
        assert "Ignoring EXAMPLE_TOOL_SERVERS__GAMMA__URL" in caplog.text

    def test_former_shorthand_is_an_ignored_variable(self, caplog: pytest.LogCaptureFixture) -> None:
        """Only HELP and SERVERS__<NAME>__... are read; generic names like <P>URL are left alone."""
        caplog.set_level(logging.DEBUG, logger="mcp_to_cli_kit.config")
        default = {"servers": {"alpha": {"url": "https://alpha.example.com/mcp"}}}

        config = load_config(
            BIN,
            default=default,
            environ={"EXAMPLE_TOOL_URL": "https://env.example.com/mcp", "EXAMPLE_TOOL_HEADERS__X_TEAM": "env"},
        ).config

        assert config.servers["alpha"].url == "https://alpha.example.com/mcp"
        assert config.servers["alpha"].headers == {}
        assert "Ignoring EXAMPLE_TOOL_URL:" in caplog.text
        assert "Ignoring EXAMPLE_TOOL_HEADERS__X_TEAM:" in caplog.text

    def test_former_shorthand_is_ignored_with_several_servers_too(self) -> None:
        config = self.load(EXAMPLE_TOOL_URL="https://env.example.com/mcp", EXAMPLE_TOOL_TIMEOUT="1")

        assert config.servers["alpha"].timeout == 10

    def test_lowercase_server_name_and_key(self) -> None:
        config = self.load(
            EXAMPLE_TOOL_SERVERS__beta_two__timeout="8", EXAMPLE_TOOL_servers__Alpha__Headers__x_team="env"
        )

        assert config.servers["beta-two"].timeout == 8
        assert config.servers["alpha"].headers["X-Team"] == "env"

    def test_lowercase_prefix(self) -> None:
        config = self.load(example_tool_servers__alpha__timeout="4", example_tool_help="lower")

        assert config.servers["alpha"].timeout == 4
        assert config.help == "lower"

    def test_lowercase_mount_key(self) -> None:
        config = self.load(EXAMPLE_TOOL_SERVERS__ALPHA__mount__group="a")

        assert config.servers["alpha"].mount == GroupMount(group="a")

    def test_lowercase_unknown_server_key_is_still_an_error(self) -> None:
        with pytest.raises(ConfigError, match="Unknown server key TOKEN"):
            self.load(example_tool_servers__alpha__token="1")

    def test_mount_group_override_drops_the_default_help(self) -> None:
        default = {
            "servers": {"alpha": {"url": "https://alpha.example.com/mcp", "mount": {"group": "a", "help": "Alpha"}}}
        }

        config = load_config(BIN, default=default, environ={"EXAMPLE_TOOL_SERVERS__ALPHA__MOUNT__GROUP": "b"}).config

        assert config.servers["alpha"].mount == GroupMount(group="b", help=None)

    def test_model_default_with_an_env_mount_override(self) -> None:
        default = CliConfig(
            servers={
                "alpha": ServerConfig(url="https://alpha.example.com/mcp", mount=GroupMount(group="a", help="Alpha"))
            }
        )

        config = load_config(BIN, default=default, environ={"EXAMPLE_TOOL_SERVERS__ALPHA__MOUNT__PREFIX": "x-"}).config

        assert config.servers["alpha"].mount == PrefixMount(prefix="x-")
        assert config.servers["alpha"].url == "https://alpha.example.com/mcp"
        assert default.servers["alpha"].mount == GroupMount(group="a", help="Alpha")

    def test_unknown_variable_is_ignored_and_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG, logger="mcp_to_cli_kit.config")

        config = self.load(EXAMPLE_TOOL_TOKEN="x", EXAMPLE_TOOL_SERVERS="y", EXAMPLE_TOOL_SERVERS__ALPHA="z")

        assert config.servers["alpha"].url == "https://alpha.example.com/mcp"
        for variable in ("EXAMPLE_TOOL_TOKEN", "EXAMPLE_TOOL_SERVERS", "EXAMPLE_TOOL_SERVERS__ALPHA"):
            assert f"Ignoring {variable}:" in caplog.text

    def test_unknown_variable_is_silent_without_debug_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)

        self.load(EXAMPLE_TOOL_TOKEN="x")

        assert caplog.text == ""

    def test_unknown_server_key_is_an_error(self) -> None:
        with pytest.raises(ConfigError, match="Unknown server key TOKEN in EXAMPLE_TOOL_SERVERS__ALPHA__TOKEN"):
            self.load(EXAMPLE_TOOL_SERVERS__ALPHA__TOKEN="1")

    def test_too_deep_is_an_error(self) -> None:
        with pytest.raises(ConfigError, match="nests deeper"):
            self.load(EXAMPLE_TOOL_SERVERS__ALPHA__URL__X="1")

    def test_other_prefixes_are_ignored(self) -> None:
        config = self.load(OTHER_TOOL_URL="x", EXAMPLE_TOOLING="y")

        assert config.servers["alpha"].url == "https://alpha.example.com/mcp"

    def test_custom_prefix(self) -> None:
        config = load_config(
            BIN, default=DEFAULT, environ={"CUSTOM_SERVERS__ALPHA__TIMEOUT": "3"}, env_prefix="CUSTOM_"
        ).config

        assert config.servers["alpha"].timeout == 3

    def test_invalid_number_names_the_sources(self) -> None:
        with pytest.raises(ConfigError) as error:
            self.load(EXAMPLE_TOOL_SERVERS__ALPHA__TIMEOUT="soon")

        assert "servers.alpha.timeout" in str(error.value)
        assert "EXAMPLE_TOOL_* environment variables" in str(error.value)


class TestModel:
    def validate(self, data: dict[str, Any]) -> CliConfig:
        return CliConfig.model_validate(data)

    def test_unknown_top_level_key(self) -> None:
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            self.validate({"servers": {"a": {"url": "https://a.example.com"}}, "colour": "red"})

    def test_unknown_server_key(self) -> None:
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            self.validate({"servers": {"a": {"url": "https://a.example.com", "token": "x"}}})

    def test_unknown_key_reported_through_load_config(self) -> None:
        write_yaml(default_config_path(BIN), {"servers": {"alpha": {"tokne": "x"}}})

        with pytest.raises(ConfigError, match=r"servers\.alpha\.tokne: Extra inputs are not permitted"):
            load_config(BIN, default=DEFAULT, environ={})

    def test_at_least_one_server(self) -> None:
        with pytest.raises(ValueError, match="servers"):
            self.validate({"servers": {}})

    def test_url_must_be_http(self) -> None:
        with pytest.raises(ValueError, match="without a scheme"):
            self.validate({"servers": {"a": {"url": "mcp.example.com"}}})

    def test_url_needs_a_host(self) -> None:
        with pytest.raises(ValueError, match="with a host"):
            self.validate({"servers": {"a": {"url": "https:///mcp"}}})

    def test_uppercase_scheme_is_accepted(self) -> None:
        assert (
            self.validate({"servers": {"a": {"url": "HTTPS://a.example.com"}}}).servers["a"].url
            == "HTTPS://a.example.com"
        )

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://user:s3cr3t-token@a.example.com/mcp?key=s3cr3t-token",
            "a.example.com/mcp?key=s3cr3t-token",
            "https://?key=s3cr3t-token",
        ],
    )
    def test_url_error_never_echoes_the_url(self, url: str) -> None:
        write_yaml(default_config_path(BIN), {"servers": {"alpha": {"url": url}}})

        with pytest.raises(ConfigError) as error:
            load_config(BIN, default=DEFAULT, environ={})

        assert "servers.alpha.url" in str(error.value)
        assert "s3cr3t" not in str(error.value)
        assert "example" not in str(error.value).split("servers.alpha.url", 1)[1]

    def test_server_name_must_be_a_safe_file_name(self) -> None:
        with pytest.raises(ValueError, match="pattern"):
            self.validate({"servers": {"../a": {"url": "https://a.example.com"}}})

    @pytest.mark.parametrize("name", ["a__b", "a-_b", "a--b", "a_", "a-", "_a"])
    def test_server_name_has_single_separators_only(self, name: str) -> None:
        with pytest.raises(ValueError, match="pattern"):
            self.validate({"servers": {name: {"url": "https://a.example.com"}}})

    @pytest.mark.parametrize("name", ["a", "a-b", "a_b", "a-b_c9", "0"])
    def test_server_name_accepts(self, name: str) -> None:
        assert list(self.validate({"servers": {name: {"url": "https://a.example.com"}}}).servers) == [name]

    def test_server_names_must_differ_beyond_dash_and_underscore(self) -> None:
        with pytest.raises(ValueError, match="duplicate server name"):
            self.validate(
                {"servers": {"a-b": {"url": "https://a.example.com"}, "a_b": {"url": "https://b.example.com"}}}
            )

    def test_duplicate_group_mounts(self) -> None:
        with pytest.raises(ValueError, match="duplicate mount group: shared"):
            self.validate(
                {
                    "servers": {
                        "a": {"url": "https://a.example.com", "mount": {"group": "shared"}},
                        "b": {"url": "https://b.example.com", "mount": {"group": "shared"}},
                    }
                }
            )

    def test_duplicate_prefix_mounts(self) -> None:
        with pytest.raises(ValueError, match="duplicate mount prefix: x-"):
            self.validate(
                {
                    "servers": {
                        "a": {"url": "https://a.example.com", "mount": {"prefix": "x-"}},
                        "b": {"url": "https://b.example.com", "mount": {"prefix": "x-"}},
                    }
                }
            )

    def test_root_tool_group_collides_with_a_group_mount(self) -> None:
        with pytest.raises(ValueError, match="duplicate top-level command name .*: search"):
            self.validate(
                {
                    "servers": {
                        "a": {"url": "https://a.example.com", "tool_groups": [{"prefix": "s_", "group": "search"}]},
                        "b": {"url": "https://b.example.com", "mount": {"group": "search"}},
                    }
                }
            )

    def test_same_tool_group_in_two_root_servers(self) -> None:
        with pytest.raises(ValueError, match="duplicate top-level command name .*: search"):
            self.validate(
                {
                    "servers": {
                        "a": {"url": "https://a.example.com", "tool_groups": [{"prefix": "s_", "group": "search"}]},
                        "b": {"url": "https://b.example.com", "tool_groups": [{"prefix": "q_", "group": "search"}]},
                    }
                }
            )

    def test_tool_group_inside_a_group_mount_does_not_collide_at_the_top(self) -> None:
        config = self.validate(
            {
                "servers": {
                    "a": {"url": "https://a.example.com", "tool_groups": [{"prefix": "s_", "group": "search"}]},
                    "b": {
                        "url": "https://b.example.com",
                        "mount": {"group": "beta"},
                        "tool_groups": [{"prefix": "s_", "group": "search"}],
                    },
                }
            }
        )

        assert list(config.servers) == ["a", "b"]

    def test_several_root_mounts_are_allowed(self) -> None:
        config = self.validate(
            {"servers": {"a": {"url": "https://a.example.com"}, "b": {"url": "https://b.example.com"}}}
        )

        assert [server.mount for server in config.servers.values()] == ["root", "root"]

    def test_unknown_mount_word(self) -> None:
        with pytest.raises(ValueError, match="mount"):
            self.validate({"servers": {"a": {"url": "https://a.example.com", "mount": "top"}}})

    def test_mount_with_both_group_and_prefix(self) -> None:
        with pytest.raises(ValueError, match="mount"):
            self.validate({"servers": {"a": {"url": "https://a.example.com", "mount": {"group": "g", "prefix": "p"}}}})

    def test_duplicate_tool_group_prefixes(self) -> None:
        with pytest.raises(ValueError, match="duplicate tool_groups prefix: x_"):
            self.validate(
                {
                    "servers": {
                        "a": {
                            "url": "https://a.example.com",
                            "tool_groups": [{"prefix": "x_", "group": "one"}, {"prefix": "x_", "group": "two"}],
                        }
                    }
                }
            )

    def test_duplicate_tool_group_names(self) -> None:
        with pytest.raises(ValueError, match="duplicate tool_groups group: one"):
            self.validate(
                {
                    "servers": {
                        "a": {
                            "url": "https://a.example.com",
                            "tool_groups": [{"prefix": "x_", "group": "one"}, {"prefix": "y_", "group": "one"}],
                        }
                    }
                }
            )

    def test_defaults(self) -> None:
        server = ServerConfig(url="https://a.example.com")

        assert server.headers == {}
        assert server.timeout == 30
        assert server.sse_read_timeout == 300
        assert server.mount == "root"
        assert server.tool_groups == []
        assert server.hidden_tools == []


class TestSaveConfig:
    def test_symlink_is_kept_and_its_target_written(self, tmp_path: Path) -> None:
        real = tmp_path / "dotfiles" / "config.yaml"
        real.parent.mkdir()
        real.write_text("servers: {}\n", encoding="utf-8")
        link = tmp_path / "config" / "config.yaml"
        link.parent.mkdir()
        link.symlink_to(real)

        written = save_config(link, {"servers": {"alpha": {"url": "https://alpha.example.com/mcp"}}})

        assert written == real.resolve()
        assert link.is_symlink()
        assert yaml.safe_load(link.read_text(encoding="utf-8")) == {
            "servers": {"alpha": {"url": "https://alpha.example.com/mcp"}}
        }
        assert stat.S_IMODE(real.stat().st_mode) == 0o600
        assert [p.name for p in link.parent.iterdir()] == ["config.yaml"]

    def test_written_file_is_private_and_loads_back(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "config.yaml"
        data = {"servers": {"alpha": {"url": "https://alpha.example.com/mcp", "headers": {"Authorization": "x"}}}}

        written = save_config(path, data)

        assert written == path.resolve()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert load_config(BIN, config_path=path, environ={}).config.servers["alpha"].headers == {"Authorization": "x"}

    def test_overwrite_tightens_permissions(self, tmp_path: Path) -> None:
        path = tmp_path / "private" / "config.yaml"
        path.parent.mkdir()
        path.write_text("servers: {}\n", encoding="utf-8")
        path.chmod(0o644)

        save_config(path, {"servers": {"alpha": {"url": "https://alpha.example.com/mcp"}}})

        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert [p.name for p in path.parent.iterdir()] == ["config.yaml"]

    def test_non_ascii_and_yaml_specials_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        data = {"help": "Příliš žluťoučký kůň: #1 [test]", "servers": {"alpha": {"url": "https://a.example.com"}}}

        save_config(path, data)

        assert "Příliš" in path.read_text(encoding="utf-8")
        assert load_config(BIN, config_path=path, environ={}).config.help == data["help"]

    def test_model_is_saved_without_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"

        save_config(path, CliConfig(servers={"alpha": ServerConfig(url="https://a.example.com")}))

        assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
            "servers": {"alpha": {"url": "https://a.example.com"}}
        }
