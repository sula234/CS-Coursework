# Believe in yourself
# Don't give up
.data
	DISPLAY: .space 524288
	DISPLAYWIDTH: .word 512
	DISPLAYHEIGHT: .word 256
	# I took the colors from here - https://htmlcolorcodes.com/color-names/
	# It's a good idea to change all the colors, so that your code 
	# will differ from other students.
	# The only requirement is - all colors must be visible on Bitmap Display
	RED: .word 0xff0000
	BLUE: .word 0x0000ff
	DEEPSKYBLUE: .word 0x00BFFF
	DEEPPINK: .word 0xFF1493
	GREEN: .word 0x00ff00
	MAROON: .word 0x800000
	ORANGE: .word 0xFFA500
	x_stretch: .float 32.0			# to stretch the values of x
	y_stretch: .float 24.0			# to stretch the values of y
	x_padding: .word 256			# to move the center
	y_padding: .word 128			# to move the center
	# some of the constants  in the formulas are already calculated
	wing_stop_y: .float -2.4619554
	ears_outer_start_x: .float 0.75
	ears_inner_start_x: .float 0.5
	zero_point_one: .float 0.1
	zero_F: .float 0.0
	zero_point_zero_one: .float 0.01	# the advised step to increment x or y, depending on the formula
	two_twenty_five: .float 2.25
	three_point_zero: .float 3.0
	four_point_zero: .float 4.0
	shoulder_range:	.float 3.0
	nine_point_zero: .float 9.0
	eight_point_zero: .float 8.0
	one_point_zero: .float 1.0
	one_point_five: .float 1.5
	minus_one_point_zero: .float -1.0
	seven_point_zero: .float 7.0
	shoulder_const_1: .float 2.71 
	shoulder_const_2: .float 1.3552619
	membrane_const_1: .float 0.0913722
	two_point_zero: .float 2.0
.text
	
	# from the cosine lab, I found that the use of temporary registers inside the main prcedure is not a good idea)

	# lw $t8 x_padding
	# lw $t9 y_padding
	# l.s $f16 x_stretch
	# l.s $f17 y_stretch
	
	lw $a2 GREEN
	jal Wing
	
	lw $a2 BLUE
	jal Ears_Outer
	
	lw $a2 RED
	jal Ears_Inner
	
	lw $a2 DEEPSKYBLUE
	jal Forehead
	
	lw $a2 DEEPPINK
	jal Shoulder
	
	lw $a2 ORANGE
	jal Membrane
	
	# this problem is nothing!!! It can't stop me!!!
	# next time give us implementaion of the BST
	li $v0 10
	syscall

Wing:
	addi $sp $sp -52
	s.s $f20 ($sp)
	s.s $f21 4($sp)
	s.s $f22 8($sp)
	s.s $f23 12($sp)
	s.s $f25 16($sp)
	s.s $f26 20($sp)
	s.s $f27 24($sp)
	s.s $f28 28($sp)
	s.s $f29 32($sp)
	s.s $f30 36($sp)
	sw $s0 40($sp)
	sw $s1 44($sp)
	sw $ra 48($sp)
	
	
	l.s $f29 three_point_zero # stop cond for loop
	 
	l.s $f30 zero_point_zero_one
	lw $s0 x_padding
	lw $s1 y_padding
	l.s $f28 one_point_zero
	l.s $f27 nine_point_zero
	l.s $f25 x_stretch
	l.s $f26 y_stretch
	l.s $f21 wing_stop_y  # f20 = x, f21 = y, $f22
	l.s $f23 seven_point_zero
	
	wing_loop:
	
	mul.s $f20 $f21 $f21 # y^2
	div.s $f20 $f20 $f27 # y^2/9
	
	sub.s $f20 $f28 $f20
	sqrt.s $f20 $f20
	mul.s $f20 $f20 $f23
	
	c.lt.s $f20 $f29
	bc1t exit_wing
	
	mul.s $f20 $f20 $f25
	mul.s $f22 $f21 $f26 # stretch x and y 
	
	mov.s $f12 $f22
	jal Float_To_Int
	move $a1 $v1 # convert y to int
	
	mov.s $f12 $f20
	jal Float_To_Int
	move $a0 $v1 # convert x to int
	
	mul $a1 $a1 -1
	add $a0 $a0 $s0
	add $a1 $a1 $s1 # set x and y relative to center
	
	jal set_pixel_color
	
	mov.s $f12 $f20
	jal Float_To_Int
	move $a0 $v1 # convert x to int
	
	
	mul $a0 $a0 -1
	add $a0 $a0 $s0
	jal set_pixel_color
	
	add.s $f21 $f21 $f30 # add 0.01 to y
	j wing_loop
	
	exit_wing:
					
	l.s $f20 ($sp)
	l.s $f21 4($sp)
	l.s $f22 8($sp)
	l.s $f23 12($sp)
	l.s $f24 16($sp)
	l.s $f25 20($sp)
	l.s $f26 24($sp)
	l.s $f27 28($sp)
	l.s $f28 32($sp)
	l.s $f29 36($sp)
	l.s $f30 40($sp)
	lw $s0 40($sp)
	lw $s1 44($sp)
	lw $ra 48($sp)
	addi $sp $sp 52
	
	jr $ra
	
Forehead:
	addi $sp $sp -56
	s.s $f20 ($sp)
	s.s $f21 4($sp)
	s.s $f22 8($sp)
	s.s $f23 12($sp)
	s.s $f24 16($sp)
	s.s $f25 20($sp)
	s.s $f26 24($sp)
	s.s $f27 28($sp)
	s.s $f28 32($sp)
	s.s $f29 36($sp)
	s.s $f30 40($sp)
	sw $s0 44($sp)
	sw $s1 48($sp)
	sw $ra 52($sp)
	
	l.s $f20 ears_inner_start_x
	l.s $f29 zero_F # condition to stop loop
	l.s $f30 zero_point_zero_one
	lw $s0 x_padding
	lw $s1 y_padding
	l.s $f25 x_stretch
	l.s $f26 y_stretch
	
	l.s $f20 ears_inner_start_x # start for x
	l.s $f21 two_twenty_five # y = 2.25
	mul.s $f21 $f21 $f26
	head_loop: 
	c.le.s $f20 $f29 
	bc1t exit_head  

	mul.s $f22 $f20 $f25
	
	mov.s $f12 $f21
	jal Float_To_Int
	move $a1 $v1 # convert y to int
	
	mov.s $f12 $f22
	jal Float_To_Int
	move $a0 $v1 # convert x to int
	
	mul $a1 $a1 -1
	add $a0 $a0 $s0
	add $a1 $a1 $s1 # set x and y relative to center
	
	jal set_pixel_color
	
	mov.s $f12 $f22
	jal Float_To_Int
	move $a0 $v1 # convert x to int
	
	
	mul $a0 $a0 -1
	add $a0 $a0 $s0
	jal set_pixel_color
	
	sub.s $f20 $f20 $f30 # add 0.01 to x
	j head_loop
	
	exit_head:
	
	l.s $f20 ($sp)
	l.s $f21 4($sp)
	l.s $f22 8($sp)
	l.s $f23 12($sp)
	l.s $f24 16($sp)
	l.s $f25 20($sp)
	l.s $f26 24($sp)
	l.s $f27 28($sp)
	l.s $f28 32($sp)
	l.s $f29 36($sp)
	l.s $f30 40($sp)
	lw $s0 44($sp)
	lw $s1 48($sp)
	lw $ra 52($sp)
	addi $sp $sp 56
	jr $ra

Membrane:
	addi $sp $sp -56
	s.s $f20 ($sp)
	s.s $f21 4($sp)
	s.s $f22 8($sp)
	s.s $f23 12($sp)
	s.s $f24 16($sp)
	s.s $f25 20($sp)
	s.s $f26 24($sp)
	s.s $f27 28($sp)
	s.s $f28 32($sp)
	s.s $f29 36($sp)
	s.s $f30 40($sp)
	sw $s0 44($sp)
	sw $s1 48($sp)
	sw $ra 52($sp)
	
	l.s $f29 four_point_zero # condition to stop loop
	l.s $f30 zero_point_zero_one
	lw $s0 x_padding
	lw $s1 y_padding
	l.s $f25 x_stretch
	l.s $f26 y_stretch
	
	l.s $f23 three_point_zero
	l.s $f24 membrane_const_1
	l.s $f27 two_point_zero
	l.s $f20 zero_F
	l.s $f16 one_point_zero
	mem_loop:

	c.le.s $f29 $f20 
	bc1t exit_mem
	
	div.s $f21 $f20 $f27 # x/2
	
	mul.s $f28 $f20 $f20 # x^2
	mul.s $f28 $f28 $f24
	
	sub.s $f21 $f21 $f28
	sub.s $f21 $f21 $f23
	
	sub.s $f28 $f20 $f27
	abs.s $f28 $f28
	sub.s $f28 $f28 $f16
	mul.s $f28 $f28 $f28
	sub.s $f28 $f16 $f28
	sqrt.s $f28 $f28
	
	add.s $f21 $f21 $f28
	
	mul.s $f22 $f20 $f25
	mul.s $f21 $f21 $f26 # stretch x and y 
	
	mov.s $f12 $f21
	jal Float_To_Int
	move $a1 $v1 # convert y to int
	
	mov.s $f12 $f22
	jal Float_To_Int
	move $a0 $v1 # convert x to int
	
	mul $a1 $a1 -1
	add $a0 $a0 $s0
	add $a1 $a1 $s1 # set x and y relative to center
	
	jal set_pixel_color
	
	mov.s $f12 $f22
	jal Float_To_Int
	move $a0 $v1 # convert x to int
	
	
	mul $a0 $a0 -1
	add $a0 $a0 $s0
	jal set_pixel_color
	
	add.s $f20 $f20 $f30 # add 0.01 to x
	j mem_loop
	
	exit_mem:
	
	l.s $f20 ($sp)
	l.s $f21 4($sp)
	l.s $f22 8($sp)
	l.s $f23 12($sp)
	l.s $f24 16($sp)
	l.s $f25 20($sp)
	l.s $f26 24($sp)
	l.s $f27 28($sp)
	l.s $f28 32($sp)
	l.s $f29 36($sp)
	l.s $f30 40($sp)
	lw $s0 44($sp)
	lw $s1 48($sp)
	lw $ra 52($sp)
	addi $sp $sp 56
	jr $ra

Shoulder:
	addi $sp $sp -56
	s.s $f20 ($sp)
	s.s $f21 4($sp)
	s.s $f22 8($sp)
	s.s $f23 12($sp)
	s.s $f24 16($sp)
	s.s $f25 20($sp)
	s.s $f26 24($sp)
	s.s $f27 28($sp)
	s.s $f28 32($sp)
	s.s $f29 36($sp)
	s.s $f30 40($sp)
	sw $s0 44($sp)
	sw $s1 48($sp)
	sw $ra 52($sp)
	
	l.s $f29 shoulder_range # condition to stop loop
	l.s $f30 zero_point_zero_one
	lw $s0 x_padding
	lw $s1 y_padding
	l.s $f25 x_stretch
	l.s $f26 y_stretch
	l.s $f23 ears_inner_start_x
	l.s $f24 one_point_five
	l.s $f27 shoulder_const_1
	l.s $f28 one_point_zero
	
	l.s $f20 one_point_zero # start for x
	shoulder_loop:
	c.le.s $f29 $f20 
	bc1t exit_shoulder 
	
	mul.s $f21 $f20 $f23
	sub.s $f21 $f24 $f21
	add.s $f21 $f21 $f27 
	
	sub.s $f16 $f20 $f28 # x-1
	mul.s $f16 $f16 $f16 # (x-1)^2
	l.s $f17 four_point_zero
	sub.s $f16 $f17 $f16 # 4 - (x-1)^2
	sqrt.s $f16 $f16 # sqrt(4 - (x-1)^2)
	l.s $f17 shoulder_const_2
	mul.s $f16 $f16 $f17
	
	sub.s $f21 $f21 $f16
	
	mul.s $f21 $f21 $f26
	mul.s $f22 $f20 $f25 # x stretch
	
	mov.s $f12 $f21
	jal Float_To_Int
	move $a1 $v1 # convert y to int
	
	mov.s $f12 $f22
	jal Float_To_Int
	move $a0 $v1 # convert x to int
	
	mul $a1 $a1 -1
	add $a0 $a0 $s0
	add $a1 $a1 $s1 # set x and y relative to center
	
	jal set_pixel_color
	
	mov.s $f12 $f22
	jal Float_To_Int
	move $a0 $v1 # convert x to int
	
	
	mul $a0 $a0 -1
	add $a0 $a0 $s0
	jal set_pixel_color
	 
	add.s $f20 $f20 $f30 # add 0.01 to x 
	j shoulder_loop
	
	exit_shoulder:
	
	
	l.s $f20 ($sp)
	l.s $f21 4($sp)
	l.s $f22 8($sp)
	l.s $f23 12($sp)
	l.s $f24 16($sp)
	l.s $f25 20($sp)
	l.s $f26 24($sp)
	l.s $f27 28($sp)
	l.s $f28 32($sp)
	l.s $f29 36($sp)
	l.s $f30 40($sp)
	lw $s0 44($sp)
	lw $s1 48($sp)
	lw $ra 52($sp)
	addi $sp $sp 56
	jr $ra

Ears_Outer:
	addi $sp $sp -56
	s.s $f20 ($sp)
	s.s $f21 4($sp)
	s.s $f22 8($sp)
	s.s $f23 12($sp)
	s.s $f24 16($sp)
	s.s $f25 20($sp)
	s.s $f26 24($sp)
	s.s $f27 28($sp)
	s.s $f28 32($sp)
	s.s $f29 36($sp)
	s.s $f30 40($sp)
	sw $s0 44($sp)
	sw $s1 48($sp)
	sw $ra 52($sp)
	
	l.s $f29 one_point_zero 
	l.s $f30 zero_point_zero_one
	lw $s0 x_padding
	lw $s1 y_padding
	l.s $f28 eight_point_zero
	l.s $f27 nine_point_zero
	l.s $f25 x_stretch
	l.s $f26 y_stretch
	l.s $f20 ears_outer_start_x  # f20 = x, f21 = y
	
	outer_loop:

	c.le.s $f29 $f20 
	bc1t exit_outer
	
	mul.s $f21 $f28 $f20
	sub.s $f21 $f27 $f21 # find y by function
	
	mul.s $f22 $f20 $f25
	mul.s $f21 $f21 $f26 # stretch x and y 
	
	mov.s $f12 $f21
	jal Float_To_Int
	move $a1 $v1 # convert y to int
	
	mov.s $f12 $f22
	jal Float_To_Int
	move $a0 $v1 # convert x to int
	
	mul $a1 $a1 -1
	add $a0 $a0 $s0
	add $a1 $a1 $s1 # set x and y relative to center
	
	jal set_pixel_color
	
	mov.s $f12 $f22
	jal Float_To_Int
	move $a0 $v1 # convert x to int
	
	
	mul $a0 $a0 -1
	add $a0 $a0 $s0
	jal set_pixel_color
	
	add.s $f20 $f20 $f30 # add 0.01 to x
	j outer_loop
	
	exit_outer:
	
	l.s $f20 ($sp)
	l.s $f21 4($sp)
	l.s $f22 8($sp)
	l.s $f23 12($sp)
	l.s $f24 16($sp)
	l.s $f25 20($sp)
	l.s $f26 24($sp)
	l.s $f27 28($sp)
	l.s $f28 32($sp)
	l.s $f29 36($sp)
	l.s $f30 40($sp)
	sw $s0 44($sp)
	sw $s1 48($sp)
	lw $ra 52($sp)
	addi $sp $sp 56
	jr $ra
	
Ears_Inner:
	addi $sp $sp -56
	s.s $f20 ($sp)
	s.s $f21 4($sp)
	s.s $f22 8($sp)
	s.s $f23 12($sp)
	s.s $f24 16($sp)
	s.s $f25 20($sp)
	s.s $f26 24($sp)
	s.s $f27 28($sp)
	s.s $f28 32($sp)
	s.s $f29 36($sp)
	s.s $f30 40($sp)
	sw $s0 44($sp)
	sw $s1 48($sp)
	sw $ra 52($sp)
	
	l.s $f29 ears_outer_start_x 
	l.s $f30 zero_point_zero_one
	lw $s0 x_padding
	lw $s1 y_padding
	l.s $f28 three_point_zero
	l.s $f25 x_stretch
	l.s $f26 y_stretch
	l.s $f20 ears_inner_start_x  # f20 = x, f21 = y
	
	inner_loop:
	
	c.le.s $f29 $f20 
	bc1t exit_inner
	
	mul.s $f21 $f28 $f20
	add.s $f21 $f29 $f21 # find y by function
	
	mul.s $f22 $f20 $f25
	mul.s $f21 $f21 $f26 # stretch x and y 
	
	mov.s $f12 $f21
	jal Float_To_Int
	move $a1 $v1 # convert y to int
	
	mov.s $f12 $f22
	jal Float_To_Int
	move $a0 $v1 # convert x to int
	
	
	mul $a1 $a1 -1
	add $a0 $a0 $s0
	add $a1 $a1 $s1 # set x and y relative to center
	
	jal set_pixel_color
	
	mov.s $f12 $f22
	jal Float_To_Int
	move $a0 $v1 # convert x to int
	
	
	mul $a0 $a0 -1
	add $a0 $a0 $s0
	jal set_pixel_color
	
	add.s $f20 $f20 $f30 # add 0.01 to x 
	j inner_loop
	
	exit_inner:
	
	
	l.s $f20 ($sp)
	l.s $f21 4($sp)
	l.s $f22 8($sp)
	l.s $f23 12($sp)
	l.s $f24 16($sp)
	l.s $f25 20($sp)
	l.s $f26 24($sp)
	l.s $f27 28($sp)
	l.s $f28 32($sp)
	l.s $f29 36($sp)
	l.s $f30 40($sp)
	lw $s0 44($sp)
	lw $s1 48($sp)
	lw $ra 52($sp)
	addi $sp $sp 56
	jr $ra
	
Float_To_Int:	# takes parameter in $f12. returns in $v1
	cvt.w.s $f12 $f12
	mfc1 $v1 $f12
	jr $ra
	
set_pixel_color:
		lw $t2, DISPLAYWIDTH
		mul $t2, $t2, $a1 # y*DISPLAYWIDTH
		add $t2, $t2, $a0 # +x
		sll $t2, $t2, 2   # *4
		la $t1, DISPLAY   # get address of DISPLAY
		add $t1, $t1, $t2 # add the calculated address of the pixel
		sw $a2, ($t1)     # write color to that pixel
		jr $ra            # return
