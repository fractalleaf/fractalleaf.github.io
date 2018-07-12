#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

# Site set-up
MARKUP = ('md', 'ipynb') # markup languages to be used

THEME = 'themes/Flex'

# PLUGINS
PLUGIN_PATHS = ['./plugins', './pelican-plugins']
PLUGINS = ['ipynb.markup', 'render_math', 'representative_image', 'related_posts']
TYPOGRIFY = True

# PATHS
PATH = 'content' # path to content directory
STATIC_PATHS = ['Algorithms/images', 'Puzzles/images', 'pdf']

# LANGUAGE AND TIMEZONE
TIMEZONE = 'Europe/Paris'
DEFAULT_LANG = 'en'

# SITE METADATA
AUTHOR = 'Søren Frimann'
SITENAME = 'Digital Ramblings'
SITETITLE = 'Digital Ramblings'
SITESUBTITLE = 'Post Hoc, Ergo Propter Hoc'
#SITEDESCRIPTION = 'Foo Bar\'s Thoughts and Writings'
SITEURL = ''

# MENU STRUCTURE
MAIN_MENU = True
# LINKS = (('About', '{filename}/pages/about.md'),)
SOCIAL = (('linkedin', 'https://www.linkedin.com/in/sfrimann/'),
          ('github', 'https://github.com/fractalleaf'))
MENUITEMS = (('Archives', '/archives.html'),
             ('Categories', '/categories.html'),
             ('Tags', '/tags.html'),)

# SYNTAX HIGHLIGHTING
PYGMENTS_STYLE = 'native'

# ARTICLE MANAGEMENT
USE_FOLDER_AS_CATEGORY = True
DEFAULT_PAGINATION = 10
DEFAULT_CATEGORY = 'Misc'

# LICENSE
CC_LICENSE = {
    'name': 'Creative Commons Attribution-ShareAlike',
    'version': '4.0',
    'slug': 'by-sa'
}
COPYRIGHT_YEAR = 2018

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
