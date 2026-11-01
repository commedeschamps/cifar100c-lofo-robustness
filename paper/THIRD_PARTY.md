# Manuscript template and bibliography attribution

## Official ELCVIA style

`elcvia.sty` is an unmodified copy of `Elcvia.sty` from the journal's official `ElcviaTemplatesV2` package, distributed by the Computer Vision Center / Universitat Autònoma de Barcelona. Its original CVC identification and date remain in the file.

- [Journal author instructions](https://elcvia.cvc.uab.cat/about/submissions)
- [Official template download](https://elcvia.cvc.uab.cat/libraryFiles/downloadPublic/1)
- [Original package README](template/README-original.txt), retained verbatim

The article uses the current LaTeX distribution's standard `article` class and `graphicx`. The official style itself is unchanged. The separately named `elcvia-submission.sty` removes only unassigned publication metadata from the title. File identities and layout details are in [template_provenance.json](template_provenance.json).

## Bibliography style

`elcvia-numeric.bst` is a modified, separately named derivative of Patrick W. Daly's `unsrtnat.bst` (1993–2007). Its original copyright and LaTeX Project Public License notice are retained. Local changes use author initials, numeric citation order and DOI-preferred links, including DOI output for miscellaneous entries. It is not an official ELCVIA bibliography style.

The [unmodified original](third_party/unsrtnat.bst) and [LPPL 1.3c text](third_party/LPPL-1.3c.txt) accompany the derivative. The original notice permits LPPL version 1 or later. These third-party terms do not select a license for the project's original research code.

Corruption-reference attribution and its Apache-2.0 license are preserved separately in the repository's [third_party directory](../third_party/NOTICE.md), including the copies associated with saved run sources.
