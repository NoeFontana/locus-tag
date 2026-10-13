/* 
 * Modern Mermaid Initialization for MkDocs Material 
 * Uses the mermaid.run() API for better control and re-initialization support.
 */
document$.subscribe(() => {
  const scheme = document.body.getAttribute("data-md-color-scheme")
  const theme = scheme === "slate" ? "dark" : "default"

  // Configure Mermaid
  mermaid.initialize({
    theme: theme,
    startOnLoad: false,
    securityLevel: "loose",
    fontFamily: "var(--md-text-font-family)",
    flowchart: {
      useMaxWidth: true,
      htmlLabels: true,
      curve: "basis"
    },
    sequence: {
      useMaxWidth: true,
      mirrorActors: false,
      bottomMargin: 20
    }
  })

  // Support for diagrams wrapped in <code> tags (common with pymdownx.superfences)
  // We extract the code content to ensure Mermaid parses it correctly.
  const diagrams = document.querySelectorAll(".mermaid")
  diagrams.forEach((diagram) => {
    const code = diagram.querySelector("code")
    if (code) {
      diagram.innerHTML = code.textContent
    }
  })

  // Run rendering
  mermaid.run({
    querySelector: '.mermaid'
  })
})
