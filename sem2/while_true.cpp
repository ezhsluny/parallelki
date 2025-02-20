#include <iostream>

int main()
{
    int cnt = 0;
    while (cnt++ != -1)
    {
    }
    return 0;
}

void never_called()
{
    std::cout << "You are screwed" << std::endl;
}