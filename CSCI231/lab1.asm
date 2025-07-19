.data 
	a: .word 11
	c: .word 22
	x: .word 35
	y: .word 15
	g: .word -23
	k: .word 7
	
.text
	lw $s0, a
	lw $s1, c
	lw $s2, x
	lw $s3, y
	lw $s4, g
	lw $s5, k
	
	add $t1, $s0, $s1
	sub $t2, $s2, $s3
	add $t3, $s0, $s4
	sub $t4, $t1, $t2
	add $t5, $t4, $t3
	
	mul $t6, $t5, $s5
	
