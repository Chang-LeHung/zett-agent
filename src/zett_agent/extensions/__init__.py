"""Built-in extensions.

Every extension is imported from its own module, for example
``from zett_agent.extensions.coding import CodingExtension``. Keeping this
package free of re-exports means importing one extension never imports the SDKs
or storage engines the others need.
"""
