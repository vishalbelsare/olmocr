import unittest
from olmocr.bench.synth.mine_html_templates import extract_html_metadata, html_to_markdown_with_frontmatter


class TestExtractHtmlMetadata(unittest.TestCase):
    def test_extract_metadata_portuguese_document(self):
        """Test metadata extraction from a Portuguese document with mixed content."""
        html_content = """
        <html lang="pt">
        <head><title>Test Document</title></head>
        <body>
            <header>Header content here</header>
            <h1>Política de Metadados</h1>
            <p>Este é um documento de teste com texto em português.</p>
            <p>Contém múltiplos parágrafos para simular conteúdo real.</p>
            <div class="image">Image placeholder 1</div>
            <p>Mais texto após a imagem.</p>
            <footer>Footer content</footer>
        </body>
        </html>
        """
        
        metadata = extract_html_metadata(html_content)
        
        # Check language extraction
        self.assertEqual(metadata['primary_language'], 'pt')
        
        # Check rotation values (always fixed)
        self.assertTrue(metadata['is_rotation_valid'])
        self.assertEqual(metadata['rotation_correction'], 0)
        
        # Check table/diagram detection
        # With 1 image (500 chars) and small text content, image ratio > 50%
        self.assertFalse(metadata['is_table'])
        self.assertTrue(metadata['is_diagram'])  # Image estimate dominates
    
    def test_extract_metadata_table_heavy_document(self):
        """Test metadata extraction from a document that is mostly tables."""
        html_content = """
        <html lang="en">
        <body>
            <p>Small intro text</p>
            <table>
                <tr><td>Cell 1</td><td>Cell 2</td><td>Cell 3</td></tr>
                <tr><td>Data A</td><td>Data B</td><td>Data C</td></tr>
                <tr><td>More data</td><td>More data</td><td>More data</td></tr>
                <tr><td>Even more data</td><td>Even more data</td><td>Even more data</td></tr>
                <tr><td>Lots of data</td><td>Lots of data</td><td>Lots of data</td></tr>
                <tr><td>Table content</td><td>Table content</td><td>Table content</td></tr>
                <tr><td>Final row</td><td>Final row</td><td>Final row</td></tr>
            </table>
        </body>
        </html>
        """
        
        metadata = extract_html_metadata(html_content)
        
        self.assertEqual(metadata['primary_language'], 'en')
        self.assertTrue(metadata['is_table'])  # Should be True as >50% is table
        self.assertFalse(metadata['is_diagram'])
    
    def test_extract_metadata_image_heavy_document(self):
        """Test metadata extraction from a document that is mostly images."""
        html_content = """
        <html lang="es">
        <body>
            <p>Brief text</p>
            <div class="image">Image 1</div>
            <div class="image">Image 2</div>
            <div class="image">Image 3</div>
            <div class="image">Image 4</div>
            <div class="image">Image 5</div>
        </body>
        </html>
        """
        
        metadata = extract_html_metadata(html_content)
        
        self.assertEqual(metadata['primary_language'], 'es')
        self.assertFalse(metadata['is_table'])
        self.assertTrue(metadata['is_diagram'])  # Should be True as >50% is images
    
    def test_extract_metadata_language_with_region(self):
        """Test that language codes with regions (e.g., pt-BR) are shortened."""
        html_content = """
        <html lang="pt-BR">
        <body>
            <p>Texto em português brasileiro</p>
        </body>
        </html>
        """
        
        metadata = extract_html_metadata(html_content)
        
        # Should convert pt-BR to pt
        self.assertEqual(metadata['primary_language'], 'pt')
    
    def test_extract_metadata_no_html_tag(self):
        """Test extraction when there's no html tag (defaults to 'en')."""
        html_content = """
        <body>
            <p>Content without html tag</p>
        </body>
        """
        
        metadata = extract_html_metadata(html_content)
        
        self.assertEqual(metadata['primary_language'], 'en')  # Should default to 'en'
    
    def test_extract_metadata_mixed_content(self):
        """Test a document with mixed content types."""
        html_content = """<!DOCTYPE html>
                <html lang="pt-BR">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Política de Metadados para Livros e Capítulos de Livro UFPA</title>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            margin: 0;
                            padding: 20px;
                            width: 725px;
                            height: 1024px;
                            box-sizing: border-box;
                        }
                        
                        .header-logos {
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            margin-bottom: 30px;
                        }
                        
                        .image {
                            border: 2px solid black;
                            background-color: #ccc;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            color: black;
                            font-weight: bold;
                        }
                        
                        .logo-left {
                            width: 120px;
                            height: 80px;
                        }
                        
                        .logo-center {
                            width: 300px;
                            height: 80px;
                        }
                        
                        .logo-right {
                            width: 120px;
                            height: 80px;
                        }
                        
                        h1 {
                            text-align: center;
                            font-weight: bold;
                            font-size: 16px;
                            margin: 20px 0;
                            text-transform: uppercase;
                        }
                        
                        .intro-text {
                            text-align: justify;
                            margin-bottom: 20px;
                            font-size: 14px;
                            line-height: 1.4;
                        }
                        
                        table {
                            width: 100%;
                            border-collapse: collapse;
                            font-size: 12px;
                        }
                        
                        th, td {
                            border: 1px solid black;
                            padding: 8px;
                            text-align: left;
                            vertical-align: middle;
                        }
                        
                        th {
                            background-color: #f0f0f0;
                            font-weight: bold;
                            text-align: center;
                        }
                        
                        .col-metadados {
                            width: 35%;
                        }
                        
                        .col-valor {
                            width: 35%;
                        }
                        
                        .col-repetitivo {
                            width: 15%;
                            text-align: center;
                        }
                        
                        .col-condicao {
                            width: 15%;
                            text-align: center;
                        }
                        
                        footer {
                            text-align: center;
                            margin-top: 20px;
                            font-size: 14px;
                            font-weight: bold;
                        }
                    </style>
                </head>
                <body>
                    <header>
                        <div class="header-logos">
                            <div class="image logo-left">Biblioteca Central UFPA</div>
                            <div class="image logo-center">LIVRO ABERTO portal do livro aberto da UFPA</div>
                            <div class="image logo-right">SIBI/UFPA</div>
                        </div>
                    </header>

                    <main>
                        <h1>Política de Metadados para Livros e Capítulos de Livro UFPA</h1>
                        
                        <p class="intro-text">
                            Essa política de metadados possui o objetivo de garantir a consistência do trabalho executado no Portal do Livro Aberto. Dessa forma, foi desenvolvido com base no esquema de metadados do Dublin Core com adaptações para a realidade brasileira e local.
                        </p>

                        <table>
                            <thead>
                                <tr>
                                    <th class="col-metadados">METADADOS</th>
                                    <th class="col-valor">VALOR</th>
                                    <th class="col-repetitivo">REPETITIVO</th>
                                    <th class="col-condicao">CONDIÇÃO</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>dc.type</td>
                                    <td>Tipo de documento</td>
                                    <td>Não</td>
                                    <td>Obrigatório</td>
                                </tr>
                                <tr>
                                    <td>dc.title</td>
                                    <td>Título e subtítulo (se houver)</td>
                                    <td>Não</td>
                                    <td>Obrigatório</td>
                                </tr>
                                <tr>
                                    <td>dc.title.alternative</td>
                                    <td>Título alternativo</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.creator</td>
                                    <td>Autor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.creator.Lattes</td>
                                    <td>URL do currículo Lattes do autor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.creator.ORCID</td>
                                    <td>ORCID do autor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.description.affiliation</td>
                                    <td>Afiliação do autor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.organizer</td>
                                    <td>Organizador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.organizerLattes</td>
                                    <td>URL do currículo Lattes do organizador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.organizerORCID</td>
                                    <td>ORCID do organizador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.description.affiliationOrganizer</td>
                                    <td>Afiliação do organizador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.coordinator</td>
                                    <td>Coordenador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.coordinatorLattes</td>
                                    <td>URL do currículo Lattes do coordenador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.coordinatorORCID</td>
                                    <td>ORCID do coordenador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.affiliationCoordinator</td>
                                    <td>Afiliação do coordenador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.editor</td>
                                    <td>Editor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.editorLattes</td>
                                    <td>URL do currículo Lattes do editor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.editorORCID</td>
                                    <td>ORCID do editor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.description.affiliationEditor</td>
                                    <td>Afiliação do editor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                            </tbody>
                        </table>
                    </main>

                    <footer>
                        <div>3</div>
                    </footer>
                </body>
                </html>
        """
        
        metadata = extract_html_metadata(html_content)
        
        self.assertEqual(metadata['primary_language'], 'pt')
        self.assertTrue(metadata['is_table'])
        self.assertFalse(metadata['is_diagram']) 
    
    def test_extract_metadata_empty_body(self):
        """Test extraction with empty or minimal content."""
        html_content = """
        <html lang="de">
        <body></body>
        </html>
        """
        
        metadata = extract_html_metadata(html_content)
        
        self.assertEqual(metadata['primary_language'], 'de')
        self.assertFalse(metadata['is_table'])
        self.assertFalse(metadata['is_diagram'])
        self.assertTrue(metadata['is_rotation_valid'])
        self.assertEqual(metadata['rotation_correction'], 0)


class TestHtmlToMarkdown(unittest.TestCase):
    def test_title_tag_excluded_from_markdown(self):
        """Test that title tags from head are not included in markdown output."""
        html_content = """
        <html lang="en">
        <head>
            <title>This Should Not Appear In Markdown</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>Main Heading</h1>
            <p>This is the body content that should appear.</p>
        </body>
        </html>
        """
        
        markdown_with_frontmatter = html_to_markdown_with_frontmatter(html_content)
        
        # Check that the title from head tag is NOT in the markdown
        self.assertNotIn("This Should Not Appear In Markdown", markdown_with_frontmatter)
        
        # Check that body content IS in the markdown
        self.assertIn("Main Heading", markdown_with_frontmatter)
        self.assertIn("This is the body content that should appear", markdown_with_frontmatter)
        
        # Check that frontmatter is present
        self.assertTrue(markdown_with_frontmatter.startswith("---"))
    
    def test_image_with_data_description(self):
        """Test that images are converted with placeholder alt text."""
        html_content = """
        <html lang="en">
        <body>
            <p>Text before image</p>
            <div class="image" data-description="A beautiful sunset over mountains">Placeholder</div>
            <p>Text after image</p>
        </body>
        </html>
        """
        
        markdown_with_frontmatter = html_to_markdown_with_frontmatter(html_content)
        
        # Check that images use the fixed placeholder alt text
        self.assertIn("![Image Placeholder]", markdown_with_frontmatter)
        
        # Check that other content is preserved
        self.assertIn("Text before image", markdown_with_frontmatter)
        self.assertIn("Text after image", markdown_with_frontmatter)
    
    def test_image_without_data_description(self):
        """Test that images without data-description use default alt text."""
        html_content = """
        <html lang="en">
        <body>
            <div class="image">Some placeholder content</div>
        </body>
        </html>
        """
        
        markdown_with_frontmatter = html_to_markdown_with_frontmatter(html_content)
        
        # Check that default alt text is used
        self.assertIn("![Image Placeholder]", markdown_with_frontmatter)
    
    def test_headers_footers_excluded(self):
        """Test that header and footer tags are excluded from markdown."""
        html_content = """
        <html lang="en">
        <body>
            <header>
                <nav>Navigation menu that should not appear</nav>
            </header>
            <main>
                <h1>Main Content</h1>
                <p>This should appear in the markdown.</p>
            </main>
            <footer>
                <p>Footer text that should not appear</p>
            </footer>
        </body>
        </html>
        """
        
        markdown_with_frontmatter = html_to_markdown_with_frontmatter(html_content)
        
        # Check that header/footer content is excluded
        self.assertNotIn("Navigation menu", markdown_with_frontmatter)
        self.assertNotIn("Footer text", markdown_with_frontmatter)
        
        # Check that main content is included
        self.assertIn("Main Content", markdown_with_frontmatter)
        self.assertIn("This should appear in the markdown", markdown_with_frontmatter)
    
    def test_no_body_tag_fallback(self):
        """Test that content is still processed when there's no body tag."""
        html_content = """
        <div>
            <h1>Content without body tag</h1>
            <p>This should still be converted.</p>
        </div>
        """
        
        markdown_with_frontmatter = html_to_markdown_with_frontmatter(html_content)
        
        # Check that content is still converted
        self.assertIn("Content without body tag", markdown_with_frontmatter)
        self.assertIn("This should still be converted", markdown_with_frontmatter)
    
    def test_removes_triple_dashes_from_content(self):
        """Test that --- at the start or end of markdown content is removed."""
        # Test with --- at the beginning
        html_content_start = """
        <html lang="en">
        <body>
            <p>---</p>
            <p>Regular content here</p>
        </body>
        </html>
        """
        
        markdown_start = html_to_markdown_with_frontmatter(html_content_start)
        lines = markdown_start.split('\n')
        
        # Check that we have FrontMatter
        self.assertEqual(lines[0], '---')
        # Check that the content doesn't start with --- after the FrontMatter ends
        frontmatter_end = next(i for i in range(1, len(lines)) if lines[i] == '---')
        content_after_frontmatter = '\n'.join(lines[frontmatter_end + 1:])
        self.assertFalse(content_after_frontmatter.strip().startswith('---'))
        
        # Test with --- at the end
        html_content_end = """
        <html lang="en">
        <body>
            <p>Regular content here</p>
            <p>---</p>
        </body>
        </html>
        """
        
        markdown_end = html_to_markdown_with_frontmatter(html_content_end)
        # Check that content doesn't end with ---
        self.assertFalse(markdown_end.rstrip().endswith('---\n---'))
        
        # Test with --- at both beginning and end
        html_content_both = """
        <html lang="en">
        <body>
            <p>---</p>
            <p>Middle content</p>
            <p>---</p>
        </body>
        </html>
        """
        
        markdown_both = html_to_markdown_with_frontmatter(html_content_both)
        lines_both = markdown_both.split('\n')
        frontmatter_end_both = next(i for i in range(1, len(lines_both)) if lines_both[i] == '---')
        content_both = '\n'.join(lines_both[frontmatter_end_both + 1:])
        
        # Content should not start or end with ---
        self.assertFalse(content_both.strip().startswith('---'))
        self.assertFalse(content_both.strip().endswith('---'))
        # But should contain "Middle content"
        self.assertIn("Middle content", content_both)


if __name__ == '__main__':
    unittest.main()