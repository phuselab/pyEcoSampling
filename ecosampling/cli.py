# """Command Line Input handler module.
# Typical usage example:
#     args = CLI().parse()
# """

# from argparse import ArgumentParser
# # from src.__version__ import __version__


# class CLI():
#     """CLI Class contains all parameters to handle arguments.
#     Set Command Line Input groups and arguments.
#     Attributes:
#         parser (:obj:`ArgumentParser`): An ArgumentParser object.
#         subparsers: Adds subparser to the parser, each one is like a
#                     standalone aplication.
#         save_path (str): Save path of the download. Defaults to './data'.
#         input_folder (str): Path where the extractor should look to files.
#                             Defaults to './data'.
#     """

#     def __init__(self):
#         """Init CLI class with default values."""
#         desc = """Data extractor of PDF documents from the Official Gazette
#                   of the Federal District, Brazil."""
#         epilog = f'© Copyright 2020, KnEDLe Team. Version {__version__}'
#         self.parser = ArgumentParser(prog="DODFMiner", description=desc,
#                                      epilog=epilog)
#         self.subparsers = self.parser.add_subparsers(dest='subparser_name')
#         self.save_path = './'

#     @classmethod
#     def _new_group(cls, name, subparser):
#         """Create new argument group.
#         Args:
#             name: Name of the group.
#             subparser: The subparser.
#         Returns:
#             The argparse group created.
#         """
#         group = subparser.add_argument_group(name)
#         return group

#     def _extract_content_parser(self):
#         """Create parser for extraction configs."""
#         self.extract_content_parser = self.subparsers.add_parser("extract")

#         group = self._new_group('Extraction Configs', self.extract_content_parser)

#         group.add_argument('-i', '--input-folder', dest='input_folder',
#                            default='./', type=str,
#                            help='Path to the PDFs folder')

#         group.add_argument('-s', '--single-file', dest='single_file', type=str,
#                            default=None,
#                            help='Path to the single file to extract')

#         group.add_argument('-t', '--type-of-extraction', dest='type_of_extr',
#                            default=None, type=str, nargs='?',
#                            choices=['pure-text', 'blocks', 'with-titles'],
#                            help="Type of text extraction")

#         group.add_argument('-a', '--act', dest='act', default='all', type=str,
#                            choices=act_choices, nargs='*',
#                            help='Which acts to extract to CSV')

#         group.add_argument('-b', '--backend', dest='backend', default='regex',
#                            type=str, choices=['regex', 'ner'],
#                            help="The backend to be used in CSV extraction")

#         group.add_argument('-c', '--committee', dest='committee', action='store_true',
#                             help="Use committee classification for acts")

#         group.add_argument('-x', '--xml', dest='xml', default=False, nargs='*',
#                             type=bool, help="Generate TeamTat XML Annotations")

#     def get_parser(self):
#         return self.parser

#     def parse(self):
#         """Create parser and parse the arguments.
#         Returns:
#             The cli arguments parsed.
#         """
#         self._download_parser()
#         self._extract_content_parser()
#         return self.parser.parse_args()
