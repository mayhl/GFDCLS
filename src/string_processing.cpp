#include "include/string_processing.h"

namespace StringProcessing
{


	void indentLines(std::string &str)
	{
		size_t start_pos = 0;
		const std::string newline = "\n";
		const std::string indent = "\t";

		// indent first line
		str.insert(0, indent);

		while ((start_pos = str.find(newline, start_pos)) != std::string::npos) {
			// Do not indent after last new line character.
			if (start_pos == str.length() - 1)
				break;
			// Indent all other lines after new line chracter.
			str.replace(start_pos, newline.length(), newline + indent);
			start_pos += newline.length(); 

		}

	}

};