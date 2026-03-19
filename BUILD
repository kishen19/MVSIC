load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",
    # This disables header-parsing analysis during generation, preventing the AssertionError
    targets = {
        "//...": "--features=-parse_headers",
    },
)
