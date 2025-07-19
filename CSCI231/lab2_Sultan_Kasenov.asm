.text
	#load 0 to data segment
	li $t0, 0
	sw $t0, 0x10010000
	
	#load 1 to data segment
	li $t1, 1
	sw $t1, 0x10010004
	
	#compute next fib value and store it
	add $t2, $t0, $t1
	sw $t2, 0x10010008
	
	#compute next fib value and store it
	add $t3, $t1, $t2
	sw $t3, 0x1001000c
	
	#compute next fib value and store it
	add $t4, $t2, $t3
	sw $t4, 0x10010010
	
	#compute next fib value and store it
	add $t5, $t3, $t4
	sw $t5, 0x10010014
	
	#compute next fib value and store it
	add $t6, $t4, $t5
	sw $t6, 0x10010018
