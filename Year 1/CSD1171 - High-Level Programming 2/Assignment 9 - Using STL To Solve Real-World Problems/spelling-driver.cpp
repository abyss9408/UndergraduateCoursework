/*******************************************************************************
 spellchecker-driver.cpp
*******************************************************************************/
#include <iostream> // cout, endl
#include <iomanip>  // setw
#include <vector>   // vector
#include <string>   // string
#include "spelling.hpp" 
using hlp2::spell_checker;

// declare identifiers private to this file in anonymous namespace
namespace {
  void test1();
  void test2();
  void test3();
  void test4();
} // end of anonymous namespace

int main(int argc, char *argv[]) {
  int test{};
  if (argc > 1) {
    test = std::stoi(argv[1]);
  }

  switch (test) {
    case 1: test1(); return 0;
    case 2: test2(); return 0;
    case 3: test3(); return 0;
    case 4: test4(); return 0;
    case 0: test1(); test2(); test3(); test4(); return 0;
  }
}

// definitions of names declared in anonymous namespace
namespace {

void TestUppercase() {
  std::vector<std::string> words = {
        "Four",      "score",    "and",     "seven",    "years", "ago",
        "our",       "fathers",  "brought", "forth",    "on",    "this",
        "continent", "a",        "new",     "NATION",   "fast123",
        "123  abc",  "Hello!!",  "",        "*&^%$#8UPPERlower"
  };

  std::cout << "TestUppercase-----------------------------------------------------\n";
  for (std::string& w : words) {
    std::cout << "Word: " << w;
    w = hlp2::string_utils::upper_case(w);
    std::cout << " (" << w << ")" << "\n";
  }
}

void TestGetinfo(std::string const& lexicon) {
  std::cout << "TestGetinfo-----------------------------------------------------\n";
  
  spell_checker sc(lexicon);
  hlp2::spell_checker::lexicon_info info;
  spell_checker::SCResult file_result = sc.get_info(info);
  if (file_result == spell_checker::scrFILE_ERR_OPEN) {
    std::cout << "Can't open " << lexicon << "\n";
    return;
  }

  std::cout << "Lexicon: "         << lexicon    << "\n";
  std::cout << "Number of words: " << info.count    << "\n";
  std::cout << "Shortest word: "   << info.shortest << " letters" << "\n";
  std::cout << "Longest word: "    << info.longest  << " letters" << "\n";
}

void TestWordsstartingwith(std::string const& lexicon, char letter) {
  std::cout << "TestWordsstartingwith-----------------------------------------------------\n";
  spell_checker sc(lexicon);
  size_t count{};
  spell_checker::SCResult file_result = sc.words_starting_with(letter, count);
  if (file_result == spell_checker::scrFILE_ERR_OPEN) {
    std::cout << "Can't open " << lexicon << "\n";
    return;
  }
  std::cout << "Lexicon: " << lexicon << "\nNumber of words starting with " 
            << letter << ": " << count << "\n";
}

void TestSpellcheck(std::string const& lexicon, char const *word) {
  std::cout << "TestSpellcheck-----------------------------------------------------\n";
  spell_checker sc(lexicon);

  spell_checker::SCResult file_result = sc.spellcheck(word);
  if (file_result == spell_checker::scrFILE_ERR_OPEN) {
    std::cout << "Can't open " << lexicon << "\n";
    return;
  }

  if (file_result == spell_checker::scrWORD_OK) {
    std::cout << "The word " << word << " is spelled correctly.\n";
  } else {
    std::cout << "The word " << word << " is misspelled.\n";
  }
}

void TestMisspelled() {
  std::vector<std::string> words {
    "Four", "score", "and", "seven", "years", "ago", "our", "fathers",
    "brought", "forth", "on", "this", "continent", "a", "new", "nation"
  };

  std::string const& lexicon = "./input/allwords.txt"; // name of the lexicon file
  std::cout << "TestMisspelled-----------------------------------------------------\n";
  spell_checker sc(lexicon);
  size_t num_misspelled{};
  std::cout << "Misspelled words: ";
  for (std::string const& w : words) {
    spell_checker::SCResult file_result = sc.spellcheck(w); //words[i]);
    if (file_result == spell_checker::scrFILE_ERR_OPEN) {
      std::cout << "Can't open " << lexicon << "\n";
      return;
    }
    if (file_result == spell_checker::scrWORD_BAD) {
      std::cout << w;
      ++num_misspelled;
    }
  }
  if (!num_misspelled) {
    std::cout << "*** None ***";
  }
  std::cout << "\n";
}

void TestSpellcheckAllwords() {
  std::vector<std::string> const words = {
    "Four", "SCORE", "and", "sevn", "years", "ago", "ar", "fawthers", "brought",
    "foarth", "on", "this", "contnent", "a", "gnu", "nashun"
  };

  std::cout << "TestSpellcheckAllwords-----------------------------------------------------\n";
  std::string const& lexicon = "./input/allwords.txt"; // name of the lexicon file
  spell_checker sc(lexicon);
  int num_misspelled = 0;
  std::cout << "Misspelled words: ";
  for (std::string const& w : words) {
    spell_checker::SCResult file_result = sc.spellcheck(w);
    if (file_result == spell_checker::scrFILE_ERR_OPEN) {
      std::cout << "Can't open " << lexicon << "\n";
      return;
    }
    if (file_result == spell_checker::scrWORD_BAD) {
      std::cout << w << " ";
      ++num_misspelled;
    }
  }
  if (!num_misspelled) {
    std::cout << "*** None ***";
  }
  std::cout << "\n";
}

void TestWordlengths(size_t max_length) {
  std::vector<size_t> lengths(max_length + 1);

  std::cout << "TestWordlengths-----------------------------------------------------\n";
  std::string const& lexicon{"./input/allwords.txt"}; // name of lexicon file
  spell_checker sc(lexicon);
  std::cout << "Lexicon: " << lexicon << "\n";
  spell_checker::SCResult  file_result = sc.word_lengths(lengths, max_length);
  if (file_result == spell_checker::scrFILE_ERR_OPEN) {
    std::cout << "Can't open " << lexicon << "\n";
    return;
  }
  size_t total{};
  for (size_t idx{1}; idx < lengths.size(); ++idx) {
    std::cout << "Number of words of length " << std::setw(2) << (idx)
              << " is " << std::setw(6) << lengths[idx] << "\n";
    total += lengths[idx];
  }
  std::cout << "Total number of words counted: " << total << "\n";
}

void TestSplit() {
  std::cout << "TestSplit-----------------------------------------------------\n";
  
  std::string words = "When in the Course of human events";
  std::cout << "String: |" << words << "|" << std::endl;
  std::vector<std::string> tokens = hlp2::string_utils::split(words);
  std::cout << "There are " << tokens.size() << " tokens\n";
  for (std::string const& s : tokens) { std::cout << "|" << s << "|" << "\n"; }
  std::cout << "----------------------------------\n";

  words = "  When    in the    Course of human events    ";
  std::cout << "String: |" << words << "|" << std::endl;
  tokens = hlp2::string_utils::split(words);
  std::cout << "There are " << tokens.size() << " tokens\n";
  for (std::string const& s : tokens) { std::cout << "|" << s << "|" << "\n"; }
  std::cout << "----------------------------------\n";

  words = "WhenintheCourseofhumanevents";
  std::cout << "String: |" << words << "|" << "\n";
  tokens = hlp2::string_utils::split(words);
  std::cout << "There are " << tokens.size() << " tokens\n";
  for (std::string const& s : tokens) { std::cout << "|" << s << "|" << "\n"; }
  std::cout << "----------------------------------\n";

  words = "";
  std::cout << "String: |" << words << "|" << "\n";
  tokens = hlp2::string_utils::split(words);
  std::cout << "There are " << tokens.size() << " tokens\n";
  for (std::string const& s : tokens) { std::cout << "|" << s << "|" << "\n"; }
  std::cout << "----------------------------------\n";

  words = "When in the Course of human events, it becomes necessary " 
          "for one people to dissolve the political bands which have "
          "connected them with another, and to assume among the powers "
          "of the earth, the separate and equal station to which the " 
          "Laws of Nature and of Nature's God entitle them, a decent "
          "respect to the opinions of mankind requires that they "
          "should declare the causes which impel them to the "
          "separation.";
  tokens = hlp2::string_utils::split(words);
  std::cout << "There are " << tokens.size() << " tokens\n";
  for (std::string const& s : tokens) { std::cout << "|" << s << "|" << "\n"; }
}

void FindAcronyms(std::vector<std::string> const& acronyms, 
                  std::string const& lexicon, size_t maxlen, bool showstrings=true) {
  spell_checker::SCResult file_result; // for file errors
  spell_checker sc(lexicon);
  for (std::string const& str : acronyms) {
    std::vector<std::string> words;
    file_result = sc.acronym_to_word(str, words, maxlen);
    if (file_result == spell_checker::scrFILE_ERR_OPEN) {
      std::cout << "Can't open " << lexicon << "\n";
      continue;
    }

    std::cout << "Acronym: " << str << ", Words (" << words.size() << "): ";
    if (showstrings) {
      for (std::string const& w : words) { std::cout << w << " "; }
    }
    std::cout << "\n";
  }
}

void TestAcronyms() {
  std::cout << "TestAcronyms----------------------------------------------------\n";
                   
  // Acronyms, simple, small lexicon
  std::vector<std::string> acronyms{"GV", "PAP", "ABC"};
  FindAcronyms(acronyms, "./input/lexicon2.txt", 0);

  // Acronyms, big lexicon, all sizes
  acronyms = {"bbq","byob","ATHF","icbm","TCBY","bsgd","imHO"};
  FindAcronyms(acronyms, "./input/allwords.txt", 0);

  // Acronyms, big lexicon, maxlen=10
  acronyms = {"ROYG", "rofl", "otoh", "asap"};
  FindAcronyms(acronyms, "./input/allwords.txt", 10);

  // Acronyms, big lexicon, lots of output
  acronyms = {"PHAT", "OTOH", "fdr", "isbn"};
  FindAcronyms(acronyms, "./input/allwords.txt", 0);
}

void test1() {
  TestUppercase(); // upper_case
  TestGetinfo("./input/lexicon.txt");  // get_info
  TestGetinfo("./input/small.txt");    // get_info
  TestGetinfo("./input/allwords.txt"); // get_info
}

void test2() {
  TestWordsstartingwith("./input/lexicon.txt", 'a');  // words_starting_with
  TestWordsstartingwith("./input/lexicon.txt", 'A');  // words_starting_with
  TestWordsstartingwith("./input/lexicon.txt", 'l');  // words_starting_with
  TestWordsstartingwith("./input/lexicon.txt", '$');  // words_starting_with
  TestWordsstartingwith("./input/lexicon.txt", 'e');  // words_starting_with
  TestWordsstartingwith("./input/allwords.txt", 'M'); // words_starting_with
  TestWordsstartingwith("./input/allwords.txt", 'q'); // words_starting_with
  TestWordsstartingwith("./input/allwords.txt", '5'); // words_starting_with
}

void test3() {
  TestSpellcheck("./input/lexicon.txt", "APPLE");            // spellcheck, correct
  TestSpellcheck("./input/lexicon.txt", "Apple");            // spellcheck, correct
  TestSpellcheck("./input/lexicon.txt", "apple");            // spellcheck, correct
  TestSpellcheck("./input/lexicon.txt", "appl");             // spellcheck, incorrect
  TestSpellcheck("./input/small.txt", "abalone");            // spellcheck, correct
  TestSpellcheck("./input/allwords.txt", "interdigitation"); // spellcheck, correct
  TestSpellcheck("./input/allwords.txt", "interdigalizing"); // spellcheck, incorrect
}

void test4() {
  TestMisspelled();         // spellcheck, all correct
  TestSpellcheckAllwords(); // spellcheck, some misspelled
  TestWordlengths(5);       // word_lengths, 1 - 5
  TestWordlengths(20);      // word_lengths, 1 - 20
  TestSplit();              // simple tokenize + larger tokenize
  TestAcronyms();           // Acronym test: small lexicon + big lexicon + big lexicon and maxlen=10
                            // Acronym test, big, lots of output
}

} // end anonymous namespace
