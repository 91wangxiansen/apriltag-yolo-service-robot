#include "Huanyu_iwdg.h"


void Huanyu_IWDG_Init(u8 prer,u16 rlr)			
{
	IWDG_WriteAccessCmd(IWDG_WriteAccess_Enable); 
	IWDG_SetPrescaler(prer); 
	IWDG_SetReload(rlr);   
	IWDG_ReloadCounter(); 
	IWDG_Enable();      
}


void Huanyu_IWDG_Feed(void)
{
	IWDG_ReloadCounter();//reload
}
