#include "include/string_processing.h"

namespace StringProcessing
{
	void indentLines(std::string &str)
	{
		// Changeable indent length to control the width of output text
		const std::string indent = "  ";
		const std::string newline = "\n";
		const std::string newline_indent = newline + indent;

		// Indent first line
		str.insert(0, indent);
		
		size_t start_pos = 0;
		// Indent after new line characters
		while ((start_pos = str.find(newline, start_pos)) != std::string::npos) 
		{
			// Do not indent after last new line character.
			if ( start_pos == str.length() - 1 ) break;

			str.replace(start_pos, newline.length(), newline_indent );
			start_pos += newline_indent.length();
		}
	}
};